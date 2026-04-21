"""
src/controlnet_wrapper.py
──────────────────────────
Thin wrapper around the diffusers ControlNet pipeline that handles:
 - Prompt preparation (positive + negative)
 - ControlNet conditioning image preparation
 - Scheduler parameter mapping
 - Seed / reproducibility management
 - Output post-processing
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ControlNetWrapper:
    """
    Wraps StableDiffusionControlNetPipeline with a cleaner interface.

    Parameters
    ----------
    pipe    : A loaded diffusers ControlNet pipeline.
    cfg     : Full config dict.
    """

    def __init__(self, pipe, cfg: dict) -> None:
        self.pipe = pipe
        self.cfg = cfg
        self._inf_cfg = cfg.get("inference", {})

    # ──────────────────────────────────────────
    # Main generation call
    # ──────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,          # preprocessed sketch (RGB)
        *,
        negative_prompt: Optional[str] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        callback=None,
    ) -> Tuple[List[Image.Image], float]:
        """
        Generate images conditioned on a sketch and text prompt.

        Returns
        -------
        (images, latency_seconds)
        """
        # Resolve parameters (use config defaults if not overridden)
        steps = num_steps or self._inf_cfg.get("num_inference_steps", 4)
        gs = guidance_scale or self._inf_cfg.get("guidance_scale", 1.5)
        cn_scale = controlnet_scale or self._inf_cfg.get(
            "controlnet_conditioning_scale", 0.9
        )
        w = width or self._inf_cfg.get("image_width", 512)
        h = height or self._inf_cfg.get("image_height", 512)
        neg_prompt = negative_prompt or self._inf_cfg.get(
            "default_negative_prompt",
            "blurry, low quality, distorted, ugly",
        )

        # Resize control image to target resolution
        control_image = control_image.convert("RGB").resize((w, h), Image.LANCZOS)

        # Reproducible seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        logger.info(
            f"Generating {num_images}x image(s) | steps={steps} gs={gs} "
            f"cn_scale={cn_scale} size={w}x{h}"
        )

        t0 = time.perf_counter()

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=control_image,
                num_inference_steps=steps,
                guidance_scale=gs,
                controlnet_conditioning_scale=float(cn_scale),
                width=w,
                height=h,
                num_images_per_prompt=num_images,
                generator=generator,
                callback_on_step_end=self._make_callback(callback, steps) if callback else None,
            )

        latency = time.perf_counter() - t0
        logger.info(f"Generation complete in {latency:.2f}s")

        return result.images, latency

    # ──────────────────────────────────────────
    # Prompt engineering helpers
    # ──────────────────────────────────────────

    @staticmethod
    def enhance_prompt(prompt: str) -> str:
        """
        Append quality booster tokens to a user prompt.
        These generic suffixes consistently improve photorealism.
        """
        boosters = (
            "highly detailed, photorealistic, sharp focus, "
            "professional photography, 8k uhd, masterpiece"
        )
        if prompt.strip().endswith(boosters.split(",")[0].strip()):
            return prompt
        return f"{prompt.strip()}, {boosters}"

    @staticmethod
    def make_negative_prompt(extras: str = "") -> str:
        base = (
            "blurry, low quality, bad anatomy, extra limbs, missing limbs, "
            "deformed, ugly, watermark, signature, text, cropped, worst quality, "
            "jpeg artifacts, oversaturated, overexposed"
        )
        return f"{base}, {extras}".strip(", ") if extras else base

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _make_callback(user_callback, total_steps: int):
        """
        Wrap user_callback(step, total, latents) for diffusers callback protocol.
        """
        def cb(pipeline, step_index, timestep, callback_kwargs):
            if user_callback:
                user_callback(step_index + 1, total_steps, callback_kwargs.get("latents"))
            return callback_kwargs
        return cb
