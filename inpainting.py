"""
src/inpainting.py
──────────────────
Sketch-guided inpainting / image editing.

The workflow is:
  1. User uploads an image.
  2. User draws strokes on top of it to indicate *where* and *what* to change.
  3. We auto-generate a mask from the strokes (see utils/mask_utils.py).
  4. We run ControlNetInpaint: the sketch guides the *content* of the edit
     while the mask tells the model *where* to apply it.
  5. We blend the inpainted region back into the original with feathered edges.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image

from utils.mask_utils import strokes_to_mask, expand_mask
from utils.image_utils import blend_images, resize_to_multiple

logger = logging.getLogger(__name__)


class SketchInpainter:
    """
    Wraps StableDiffusionControlNetInpaintPipeline for sketch-guided editing.

    Parameters
    ----------
    pipe    : A loaded diffusers ControlNetInpaint pipeline.
    cfg     : Full config dict.
    """

    def __init__(self, pipe, cfg: dict) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self._inf_cfg = cfg.get("inference", {})
        self._edit_cfg = cfg.get("editing", {})

    # ──────────────────────────────────────────
    # Main editing call
    # ──────────────────────────────────────────

    def edit(
        self,
        original_image: Image.Image,
        sketch_overlay: Image.Image,         # strokes drawn by user on top of image
        prompt: str,
        *,
        mask: Optional[Image.Image] = None,  # provide a manual mask or auto-derive
        mask_strategy: str = "dilate",
        negative_prompt: Optional[str] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_scale: Optional[float] = None,
        strength: Optional[float] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        blend_result: bool = True,
    ) -> Tuple[List[Image.Image], float]:
        """
        Edit `original_image` guided by `sketch_overlay` and `prompt`.

        Parameters
        ----------
        original_image  : The source image to edit (PIL RGB).
        sketch_overlay  : User's strokes (black on white, or dark on light).
                          Should be the same size as original_image.
        prompt          : Describes the desired new content in the edit region.
        mask            : Optional manual inpainting mask (white = edit area).
                          If None, derived automatically from sketch strokes.
        mask_strategy   : "dilate" | "flood_fill" | "convex_hull" | "bounding_box"
        strength        : How aggressively to repaint (0 = keep, 1 = full repaint).
        blend_result    : If True, feather the result back into the original.

        Returns
        -------
        (edited_images, latency_seconds)
        """
        # Resolve config defaults
        steps = num_steps or self._inf_cfg.get("num_inference_steps", 4)
        gs = guidance_scale or self._inf_cfg.get("guidance_scale", 1.5)
        cn_scale = controlnet_scale or self._inf_cfg.get(
            "controlnet_conditioning_scale", 0.9
        )
        inpaint_strength = strength or self._edit_cfg.get("inpaint_strength", 0.85)
        mask_blur = self._edit_cfg.get("mask_blur_radius", 8)
        mask_dilation = self._edit_cfg.get("mask_dilation", 12)
        neg_prompt = negative_prompt or self._inf_cfg.get(
            "default_negative_prompt",
            "blurry, low quality, distorted",
        )

        # Resize to a model-friendly size (multiple of 8)
        target_w = self._inf_cfg.get("image_width", 512)
        target_h = self._inf_cfg.get("image_height", 512)
        orig_size = original_image.size

        image_resized = original_image.convert("RGB").resize(
            (target_w, target_h), Image.LANCZOS
        )
        sketch_resized = sketch_overlay.convert("L").resize(
            (target_w, target_h), Image.LANCZOS
        )

        # Derive inpainting mask if not provided
        if mask is None:
            logger.info(f"Auto-deriving mask (strategy={mask_strategy})")
            mask = strokes_to_mask(
                sketch_resized,
                strategy=mask_strategy,
                dilation=mask_dilation,
                blur=mask_blur,
            )
        else:
            mask = mask.convert("L").resize((target_w, target_h), Image.LANCZOS)

        # Expand mask slightly to avoid sharp boundaries
        mask = expand_mask(mask, expand_pixels=5, feather=mask_blur)

        # ControlNet condition image = cleaned sketch
        control_img = sketch_resized.convert("RGB")

        # Seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        logger.info(
            f"Inpainting | steps={steps} gs={gs} cn={cn_scale} strength={inpaint_strength}"
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=image_resized,
                mask_image=mask,
                control_image=control_img,
                num_inference_steps=steps,
                guidance_scale=gs,
                controlnet_conditioning_scale=float(cn_scale),
                strength=inpaint_strength,
                width=target_w,
                height=target_h,
                num_images_per_prompt=num_images,
                generator=generator,
            )
        latency = time.perf_counter() - t0
        logger.info(f"Inpainting complete in {latency:.2f}s")

        edited_images = result.images

        # Blend edited region back into original (restores non-masked areas perfectly)
        if blend_result:
            edited_images = [
                blend_images(
                    original_image.resize((target_w, target_h), Image.LANCZOS),
                    edited_img,
                    mask,
                    blur_mask=mask_blur,
                )
                for edited_img in edited_images
            ]
            # Optionally resize back to original dimensions
            edited_images = [
                img.resize(orig_size, Image.LANCZOS) for img in edited_images
            ]

        return edited_images, latency

    # ──────────────────────────────────────────
    # Convenience: get a preview of the mask before running inference
    # ──────────────────────────────────────────

    def preview_mask(
        self,
        original_image: Image.Image,
        sketch_overlay: Image.Image,
        strategy: str = "dilate",
    ) -> Image.Image:
        """Return a visual preview of the auto-derived mask overlaid on the image."""
        from utils.mask_utils import visualise_mask

        mask = strokes_to_mask(
            sketch_overlay.convert("L"),
            strategy=strategy,
            dilation=self._edit_cfg.get("mask_dilation", 12),
            blur=self._edit_cfg.get("mask_blur_radius", 8),
        )
        return visualise_mask(original_image, mask)
