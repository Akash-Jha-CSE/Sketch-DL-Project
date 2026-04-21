"""
src/pipeline.py
────────────────
Top-level orchestrator that ties together:
  - ModelManager        (lazy model loading)
  - SketchPreprocessor  (sketch_utils)
  - ControlNetWrapper   (generation)
  - SketchInpainter     (editing / inpainting)
  - LatencyTracker      (performance monitoring)

This is the single class the UI calls — all heavy wiring is hidden here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml
from PIL import Image

from models.model_manager import ModelManager
from src.controlnet_wrapper import ControlNetWrapper
from src.inpainting import SketchInpainter
from src.optimizer import LatencyTracker, recommend_settings, free_vram
from utils.sketch_utils import sketch_to_controlnet_input
from utils.image_utils import make_grid, save_image

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class SketchPipeline:
    """
    Unified pipeline for:
      1. Sketch-to-image generation  (generate)
      2. Sketch-guided editing       (edit)

    Example
    -------
    >>> pipe = SketchPipeline()
    >>> images, latency = pipe.generate(
    ...     sketch=sketch_pil,
    ...     prompt="a golden retriever sitting on grass",
    ... )
    >>> edited, latency = pipe.edit(
    ...     original=photo_pil,
    ...     sketch_overlay=strokes_pil,
    ...     prompt="add a window to the wall",
    ... )
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.cfg = load_config(config_path)
        self._setup_logging()
        self._setup_output_dirs()

        self._model_manager: Optional[ModelManager] = None
        self._gen_wrapper: Optional[ControlNetWrapper] = None
        self._inpainter: Optional[SketchInpainter] = None

        self._gen_latency = LatencyTracker()
        self._edit_latency = LatencyTracker()

        logger.info("SketchPipeline initialised (models load lazily on first call)")

    # ──────────────────────────────────────────
    # Lazy initialisation
    # ──────────────────────────────────────────

    def _ensure_models_loaded(self, mode: str = "generate") -> None:
        if self._model_manager is None:
            self._model_manager = ModelManager(self.cfg)
            hw_rec = recommend_settings(self._model_manager.device)
            logger.info(f"Hardware recommendation: {hw_rec.get('note', '')}")

        if mode == "generate" and self._gen_wrapper is None:
            logger.info("Loading generation pipeline …")
            pipe = self._model_manager.get_generation_pipeline()
            self._gen_wrapper = ControlNetWrapper(pipe, self.cfg)
            logger.info("Generation pipeline ready ✓")

        if mode == "edit" and self._inpainter is None:
            logger.info("Loading inpainting pipeline …")
            pipe = self._model_manager.get_inpainting_pipeline()
            self._inpainter = SketchInpainter(pipe, self.cfg)
            logger.info("Inpainting pipeline ready ✓")

    # ──────────────────────────────────────────
    # Public: generate
    # ──────────────────────────────────────────

    def generate(
        self,
        sketch: Image.Image,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_scale: Optional[float] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
        save_outputs: bool = False,
    ) -> Tuple[List[Image.Image], float]:
        """
        Generate image(s) from a sketch + text prompt.

        Returns
        -------
        (images, latency_seconds)
        """
        self._ensure_models_loaded("generate")

        inf_cfg = self.cfg.get("inference", {})
        w = inf_cfg.get("image_width", 512)
        h = inf_cfg.get("image_height", 512)

        # Preprocess sketch for ControlNet
        control_img = sketch_to_controlnet_input(
            sketch,
            target_size=(w, h),
            cfg=self.cfg.get("sketch_preprocessing", {}),
        )

        # Optionally enhance the prompt
        if enhance_prompt:
            prompt = ControlNetWrapper.enhance_prompt(prompt)
            logger.debug(f"Enhanced prompt: {prompt}")

        images, latency = self._gen_wrapper.generate(
            prompt=prompt,
            control_image=control_img,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            width=w,
            height=h,
            num_images=num_images,
            seed=seed,
        )

        self._gen_latency.record(latency)

        if save_outputs:
            self._save_batch(images, prefix="gen")

        logger.info(
            f"generate() | latency={latency:.2f}s | "
            f"rolling_mean={self._gen_latency.mean:.2f}s"
        )
        return images, latency

    # ──────────────────────────────────────────
    # Public: edit
    # ──────────────────────────────────────────

    def edit(
        self,
        original: Image.Image,
        sketch_overlay: Image.Image,
        prompt: str,
        *,
        mask: Optional[Image.Image] = None,
        mask_strategy: str = "dilate",
        negative_prompt: Optional[str] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_scale: Optional[float] = None,
        strength: Optional[float] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        blend_result: bool = True,
        save_outputs: bool = False,
    ) -> Tuple[List[Image.Image], float]:
        """
        Edit an existing image guided by sketch strokes and a text prompt.

        Returns
        -------
        (edited_images, latency_seconds)
        """
        self._ensure_models_loaded("edit")

        images, latency = self._inpainter.edit(
            original_image=original,
            sketch_overlay=sketch_overlay,
            prompt=prompt,
            mask=mask,
            mask_strategy=mask_strategy,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            strength=strength,
            num_images=num_images,
            seed=seed,
            blend_result=blend_result,
        )

        self._edit_latency.record(latency)

        if save_outputs:
            self._save_batch(images, prefix="edit")

        logger.info(
            f"edit() | latency={latency:.2f}s | "
            f"rolling_mean={self._edit_latency.mean:.2f}s"
        )
        return images, latency

    # ──────────────────────────────────────────
    # Public: preview mask (UI helper)
    # ──────────────────────────────────────────

    def preview_edit_mask(
        self,
        original: Image.Image,
        sketch_overlay: Image.Image,
        strategy: str = "dilate",
    ) -> Image.Image:
        """
        Return a visual preview of the mask that would be used for editing.
        Helps users verify the region before committing to a full generation.
        """
        self._ensure_models_loaded("edit")
        return self._inpainter.preview_mask(original, sketch_overlay, strategy)

    # ──────────────────────────────────────────
    # Performance stats
    # ──────────────────────────────────────────

    def latency_stats(self) -> dict:
        return {
            "generation": self._gen_latency.summary(),
            "editing": self._edit_latency.summary(),
        }

    # ──────────────────────────────────────────
    # Memory management
    # ──────────────────────────────────────────

    def unload_models(self, mode: str = "all") -> None:
        if self._model_manager:
            if mode in ("generate", "all") and self._gen_wrapper:
                self._model_manager.unload("generation")
                self._gen_wrapper = None
            if mode in ("edit", "all") and self._inpainter:
                self._model_manager.unload("inpainting")
                self._inpainter = None
        free_vram()

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _save_batch(self, images: List[Image.Image], prefix: str) -> None:
        out_dir = Path(self.cfg.get("paths", {}).get("output_dir", "./outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        ts = int(time.time())
        for i, img in enumerate(images):
            save_image(img, out_dir / f"{prefix}_{ts}_{i:02d}.png")
        if len(images) > 1:
            grid = make_grid(images, cols=min(4, len(images)))
            save_image(grid, out_dir / f"{prefix}_{ts}_grid.png")

    def _setup_logging(self) -> None:
        level = self.cfg.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    def _setup_output_dirs(self) -> None:
        for key in ("output_dir", "tmp_dir"):
            d = self.cfg.get("paths", {}).get(key)
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)
