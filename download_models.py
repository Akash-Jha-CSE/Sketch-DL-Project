"""
scripts/download_models.py
───────────────────────────
One-shot script to download all required model weights from Hugging Face Hub
into the local model cache.  Run this before launching the UI.

Usage:
    python scripts/download_models.py [--cache-dir ./model_cache] [--skip-lcm]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "description": "Stable Diffusion 1.5 (base generation model)",
    },
    {
        "id": "runwayml/stable-diffusion-inpainting",
        "description": "Stable Diffusion 1.5 Inpainting (editing mode)",
    },
    {
        "id": "lllyasviel/control_v11p_sd15_scribble",
        "description": "ControlNet Scribble (sketch conditioning)",
    },
    {
        "id": "latent-consistency/lcm-lora-sdv1-5",
        "description": "LCM-LoRA (4-step fast inference adapter)",
        "flag": "lcm",
    },
]


def download_all(cache_dir: str, skip_flags: set) -> None:
    try:
        from huggingface_hub import snapshot_download
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionControlNetInpaintPipeline,
        )
    except ImportError:
        logger.error(
            "Required packages missing. Run:\n"
            "  pip install diffusers transformers huggingface-hub"
        )
        sys.exit(1)

    import torch
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        if m.get("flag") in skip_flags:
            logger.info(f"Skipping {m['id']} (--skip-{m['flag']} flag set)")
            continue

        logger.info(f"Downloading: {m['description']}")
        logger.info(f"  Model ID : {m['id']}")
        try:
            snapshot_download(
                repo_id=m["id"],
                cache_dir=cache_dir,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            )
            logger.info(f"  ✓ Done: {m['id']}")
        except Exception as exc:
            logger.error(f"  ✗ Failed to download {m['id']}: {exc}")

    logger.info("\nAll downloads complete.")
    logger.info(f"Weights cached in: {Path(cache_dir).resolve()}")
    logger.info("You can now launch the UI with:  python ui/app.py")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--cache-dir", default="./model_cache",
        help="Directory to store downloaded model weights (default: ./model_cache)"
    )
    parser.add_argument(
        "--skip-lcm", action="store_true",
        help="Skip downloading LCM-LoRA (standard 20-step SD will be used instead)"
    )
    args = parser.parse_args()

    skip = set()
    if args.skip_lcm:
        skip.add("lcm")

    download_all(args.cache_dir, skip)


if __name__ == "__main__":
    main()
