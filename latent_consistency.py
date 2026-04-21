"""
src/latent_consistency.py
──────────────────────────
Latent Consistency Model (LCM) inference path.

LCM removes the need for classifier-free guidance and dramatically reduces
the number of denoising steps required (from ~20 down to 4–8), giving
near real-time generation on consumer GPUs.

Two modes are supported:
1. LCM-LoRA fused into an existing SD1.5 / ControlNet pipeline
   (handled in model_manager.py — this file is the high-level API).
2. Native LCM model (lcm-sdv1-5) — faster still but no ControlNet support.
   Useful for a quick text-only preview or a baseline reference image.

Reference:
  Luo et al. 2023 "Latent Consistency Models: Synthesizing High-Resolution
  Images with Few-Step Inference"  https://arxiv.org/abs/2310.04378
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class LCMFastGenerator:
    """
    Thin wrapper for native LCM (no ControlNet) text-to-image generation.
    Useful as a 'preview' generation path when low latency is critical.

    Parameters
    ----------
    cfg      : Full config dict.
    device   : torch.device to use.
    dtype    : torch.float16 or torch.float32.
    """

    LCM_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"

    def __init__(self, cfg: dict, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self._pipe = None

    # ──────────────────────────────────────────
    # Lazy initialisation
    # ──────────────────────────────────────────

    def _load_pipeline(self):
        if self._pipe is not None:
            return
        try:
            from diffusers import DiffusionPipeline, LCMScheduler
        except ImportError:
            raise RuntimeError("diffusers not installed.")

        cache_dir = self.cfg.get("paths", {}).get("model_cache_dir", "./model_cache")
        logger.info(f"Loading native LCM: {self.LCM_MODEL_ID}")

        pipe = DiffusionPipeline.from_pretrained(
            self.LCM_MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None,
            cache_dir=cache_dir,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        # Optional optimisations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_attention_slicing()

        self._pipe = pipe
        logger.info("Native LCM pipeline ready ✓")

    # ──────────────────────────────────────────
    # Generate
    # ──────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_steps: int = 4,
        guidance_scale: float = 1.0,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[List[Image.Image], float]:
        """
        Fast text-to-image generation using native LCM (no sketch conditioning).
        Typical latency: ~0.3–0.5 s on A10 GPU.

        Returns
        -------
        (images, latency_seconds)
        """
        self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            )
        latency = time.perf_counter() - t0
        logger.info(f"LCM generation: {latency:.2f}s  steps={num_steps}")
        return result.images, latency

    def unload(self):
        """Release VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            import gc
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────
# SDXL-Turbo path (1-step, img2img only — no ControlNet)
# Used for very-fast stylised previews.
# ──────────────────────────────────────────────────────────────

class SDXLTurboGenerator:
    """
    SDXL-Turbo: 1–4 step text-guided image-to-image generation.
    Can be used to rapidly stylise a sketch by treating the (coloured)
    sketch as the img2img starting point.

    Reference:
      Sauer et al. 2023  "Adversarial Diffusion Distillation"
      https://arxiv.org/abs/2311.17042
    """

    MODEL_ID = "stabilityai/sdxl-turbo"

    def __init__(self, cfg: dict, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is not None:
            return
        try:
            from diffusers import AutoPipelineForImage2Image
        except ImportError:
            raise RuntimeError("diffusers not installed.")

        cache_dir = self.cfg.get("paths", {}).get("model_cache_dir", "./model_cache")
        logger.info(f"Loading SDXL-Turbo: {self.MODEL_ID}")
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            cache_dir=cache_dir,
        ).to(self.device)
        logger.info("SDXL-Turbo ready ✓")

    def generate(
        self,
        prompt: str,
        init_image: Image.Image,
        *,
        strength: float = 0.5,
        num_steps: int = 2,
        guidance_scale: float = 0.0,       # SDXL-Turbo uses 0 guidance
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[List[Image.Image], float]:
        """
        1–4 step image-to-image with SDXL-Turbo.
        Typical latency: ~0.5 s on A10 GPU.
        """
        self._load_pipeline()

        init_image = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                generator=generator,
            )
        latency = time.perf_counter() - t0
        logger.info(f"SDXL-Turbo generation: {latency:.2f}s")
        return result.images, latency
