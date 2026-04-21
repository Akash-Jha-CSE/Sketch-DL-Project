"""
scripts/benchmark.py
─────────────────────
Measures end-to-end generation and editing latency across different
configurations (step counts, image sizes, precision modes).

Usage:
    python scripts/benchmark.py [--config configs/config.yaml] [--n 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def make_dummy_sketch(size: int = 512):
    """Create a simple synthetic sketch (circle + lines) as a PIL image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    # Draw a circle
    draw.ellipse(
        [(size // 4, size // 4), (3 * size // 4, 3 * size // 4)],
        outline="black", width=4
    )
    # Draw some lines (simulating body / legs)
    cx, cy = size // 2, 3 * size // 4
    draw.line([(cx, cy), (cx - 60, size - 20)], fill="black", width=3)
    draw.line([(cx, cy), (cx + 60, size - 20)], fill="black", width=3)
    return img


def run_benchmark(config_path: str, n_runs: int = 5) -> None:
    from src.pipeline import SketchPipeline, load_config
    import torch

    cfg = load_config(config_path)

    # Hardware info
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {dev_name}  ({vram:.0f} GB VRAM)")
    else:
        logger.info("Running on CPU")

    sketch = make_dummy_sketch(512)
    prompt = "a golden retriever sitting on green grass"

    pipe = SketchPipeline(config_path)

    configs_to_test = [
        {"label": "LCM 4-step  512px", "steps": 4,  "size": 512, "gs": 1.5},
        {"label": "LCM 8-step  512px", "steps": 8,  "size": 512, "gs": 1.5},
        {"label": "Std  20-step 512px", "steps": 20, "size": 512, "gs": 7.5},
        {"label": "LCM 4-step  384px", "steps": 4,  "size": 384, "gs": 1.5},
    ]

    results = []

    for test_cfg in configs_to_test:
        label = test_cfg["label"]
        logger.info(f"\n{'─'*50}")
        logger.info(f"Benchmarking: {label}")

        times = []
        for i in range(n_runs):
            _, latency = pipe.generate(
                sketch=sketch,
                prompt=prompt,
                num_steps=test_cfg["steps"],
                guidance_scale=test_cfg["gs"],
                num_images=1,
                enhance_prompt=False,
            )
            times.append(latency)
            logger.info(f"  Run {i+1}/{n_runs}: {latency:.3f}s")

        mean_t = np.mean(times)
        p95_t = np.percentile(times, 95)
        min_t = min(times)

        logger.info(f"  → Mean={mean_t:.3f}s  P95={p95_t:.3f}s  Min={min_t:.3f}s")
        results.append({
            "config": label,
            "steps": test_cfg["steps"],
            "size": test_cfg["size"],
            "mean_s": round(mean_t, 3),
            "p95_s": round(p95_t, 3),
            "min_s": round(min_t, 3),
            "n_runs": n_runs,
        })

    # Save results
    out_path = Path("./outputs/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Config':<25} {'Steps':>6} {'Mean(s)':>9} {'P95(s)':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['config']:<25} {r['steps']:>6} {r['mean_s']:>9.3f} {r['p95_s']:>8.3f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n", type=int, default=5, help="Number of runs per config")
    args = parser.parse_args()
    run_benchmark(args.config, args.n)
