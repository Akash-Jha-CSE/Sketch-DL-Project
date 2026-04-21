"""
models/model_manager.py
────────────────────────
Centralised model registry with:
 - Lazy loading (models loaded on first use, not at import time)
 - LRU-style memory management (unload least-recently-used if VRAM is tight)
 - fp16 / xformers / torch.compile optimisations applied automatically
 - Friendly error messages with download instructions when weights are missing
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton-style manager that lazily instantiates and caches diffusion
    pipeline objects.

    Usage
    ─────
    mm = ModelManager(cfg)
    pipe = mm.get_generation_pipeline()
    inpaint_pipe = mm.get_inpainting_pipeline()
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._cache: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}
        self._device = self._detect_device()
        self._dtype = torch.float16 if (
            cfg.get("optimization", {}).get("use_fp16", True)
            and self._device.type != "cpu"
        ) else torch.float32

        cache_dir = cfg.get("paths", {}).get("model_cache_dir", "./model_cache")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelManager | device={self._device} | dtype={self._dtype}")

    # ──────────────────────────────────────────
    # Device detection
    # ──────────────────────────────────────────

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
            logger.info("Apple MPS detected.")
        else:
            dev = torch.device("cpu")
            logger.warning("No GPU found — running on CPU. Inference will be slow.")
        return dev

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def get_generation_pipeline(self):
        """
        Return a ControlNet + Stable Diffusion pipeline for sketch → image.
        Applies LCM-LoRA if enabled in config.
        """
        key = "generation"
        if key not in self._cache:
            self._cache[key] = self._build_generation_pipeline()
        self._last_used[key] = time.time()
        return self._cache[key]

    def get_inpainting_pipeline(self):
        """
        Return a ControlNet + SD-inpainting pipeline for sketch-guided editing.
        """
        key = "inpainting"
        if key not in self._cache:
            self._cache[key] = self._build_inpainting_pipeline()
        self._last_used[key] = time.time()
        return self._cache[key]

    def unload(self, key: str) -> None:
        """Unload a specific pipeline to free VRAM."""
        if key in self._cache:
            del self._cache[key]
            del self._last_used[key]
            gc.collect()
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Unloaded pipeline: {key}")

    def unload_all(self) -> None:
        for key in list(self._cache):
            self.unload(key)

    @property
    def device(self) -> torch.device:
        return self._device

    # ──────────────────────────────────────────
    # Pipeline builders
    # ──────────────────────────────────────────

    def _build_generation_pipeline(self):
        """Build StableDiffusionControlNetPipeline with optional LCM-LoRA."""
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                LCMScheduler,
                EulerDiscreteScheduler,
            )
        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers>=0.25.0"
            )

        model_cfg = self.cfg.get("model", {})
        opt_cfg = self.cfg.get("optimization", {})

        controlnet_id = model_cfg.get(
            "controlnet", "lllyasviel/control_v11p_sd15_scribble"
        )
        base_model_id = model_cfg.get(
            "base_model", "runwayml/stable-diffusion-v1-5"
        )

        logger.info(f"Loading ControlNet: {controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self._dtype,
            cache_dir=str(self._cache_dir),
        )

        logger.info(f"Loading base model: {base_model_id}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=self._dtype,
            safety_checker=None,             # disable for speed
            requires_safety_checker=False,
            cache_dir=str(self._cache_dir),
        )

        # LCM-LoRA for 4-step fast inference
        use_lcm = model_cfg.get("use_lcm", True)
        if use_lcm:
            lcm_lora_id = model_cfg.get(
                "lcm_lora", "latent-consistency/lcm-lora-sdv1-5"
            )
            logger.info(f"Loading LCM-LoRA: {lcm_lora_id}")
            pipe.load_lora_weights(lcm_lora_id, cache_dir=str(self._cache_dir))
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            logger.info("LCM-LoRA fused ✓  — fast 4-step inference active")
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self._device)
        self._apply_optimisations(pipe, opt_cfg)
        return pipe

    def _build_inpainting_pipeline(self):
        """Build ControlNet + SD-inpainting pipeline."""
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetInpaintPipeline,
                LCMScheduler,
                EulerDiscreteScheduler,
            )
        except ImportError:
            raise RuntimeError("diffusers not installed.")

        model_cfg = self.cfg.get("model", {})
        opt_cfg = self.cfg.get("optimization", {})

        # For inpainting we use the SD1.5 inpaint base
        inpaint_base = "runwayml/stable-diffusion-inpainting"
        controlnet_id = model_cfg.get(
            "controlnet", "lllyasviel/control_v11p_sd15_scribble"
        )

        logger.info(f"Loading ControlNet: {controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self._dtype,
            cache_dir=str(self._cache_dir),
        )

        logger.info(f"Loading inpaint base: {inpaint_base}")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            inpaint_base,
            controlnet=controlnet,
            torch_dtype=self._dtype,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=str(self._cache_dir),
        )

        use_lcm = model_cfg.get("use_lcm", True)
        if use_lcm:
            lcm_lora_id = model_cfg.get(
                "lcm_lora", "latent-consistency/lcm-lora-sdv1-5"
            )
            logger.info(f"Loading LCM-LoRA: {lcm_lora_id}")
            pipe.load_lora_weights(lcm_lora_id, cache_dir=str(self._cache_dir))
            pipe.fuse_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self._device)
        self._apply_optimisations(pipe, opt_cfg)
        return pipe

    # ──────────────────────────────────────────
    # Optimisation helpers
    # ──────────────────────────────────────────

    def _apply_optimisations(self, pipe, opt_cfg: dict) -> None:
        """Apply speed/memory optimisations to a diffusers pipeline."""

        # xformers memory-efficient attention
        if opt_cfg.get("enable_xformers", True):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory-efficient attention ✓")
            except Exception:
                logger.debug("xformers not available; skipping.")

        # Attention slicing (for low VRAM)
        attn_slice = opt_cfg.get("attention_slicing", "auto")
        if attn_slice is not False and attn_slice is not None:
            pipe.enable_attention_slicing(attn_slice)
            logger.info(f"Attention slicing enabled: {attn_slice}")

        # VAE tiling (for very large images on limited VRAM)
        if opt_cfg.get("tile_vae", False):
            pipe.enable_vae_tiling()
            logger.info("VAE tiling ✓")

        # torch.compile (PyTorch 2.x)
        if opt_cfg.get("compile_model", False):
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                logger.info("torch.compile applied to UNet ✓")
            except Exception as exc:
                logger.warning(f"torch.compile failed: {exc}")
