"""
src/sketch_processor.py
────────────────────────
High-level SketchProcessor class that combines all pre-processing steps
into a single configurable object used by the pipeline.

This is the single entry-point for transforming raw user input
(canvas drawing, uploaded image, camera snap) into a ControlNet-ready
conditioning image.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PIL import Image

from utils.sketch_utils import (
    preprocess_sketch,
    canny_from_image,
    hed_from_image,
    sketch_to_controlnet_input,
    augment_sketch,
)

logger = logging.getLogger(__name__)


class SketchProcessor:
    """
    Configurable sketch preprocessing stage.

    Parameters
    ----------
    cfg : The `sketch_preprocessing` sub-section of the main config dict.
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or {}

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def process(
        self,
        sketch: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Full pipeline: denoise → threshold → invert → resize → RGB.
        Returns an RGB image ready for ControlNet.
        """
        return sketch_to_controlnet_input(sketch, target_size=target_size, cfg=self.cfg)

    def from_photo(
        self,
        photo: Image.Image,
        method: str = "canny",
        target_size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Extract edges from a photograph to use as a sketch proxy.

        Parameters
        ----------
        method : "canny" | "hed"
        """
        if method == "canny":
            edge = canny_from_image(photo, target_size=target_size)
        elif method == "hed":
            edge = hed_from_image(photo, target_size=target_size)
        else:
            raise ValueError(f"Unknown edge method: {method!r}")
        return edge.convert("RGB")

    def augment(self, sketch: Image.Image) -> Image.Image:
        """Apply random augmentations (for training data prep)."""
        return augment_sketch(sketch.convert("L"))

    # ──────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────

    def analyse_sketch_quality(self, sketch: Image.Image) -> dict:
        """
        Return a simple quality report for a sketch.
        Helps identify if a sketch is too sparse/noisy for good conditioning.
        """
        import numpy as np
        arr = np.array(sketch.convert("L"))
        stroke_pixels = (arr < 128).sum()
        total_pixels = arr.size
        coverage = stroke_pixels / total_pixels

        quality = "good"
        notes = []
        if coverage < 0.01:
            quality = "too sparse"
            notes.append("Very few strokes detected. Draw more to guide generation.")
        elif coverage > 0.70:
            quality = "too dense"
            notes.append("Sketch is very dark/filled. ControlNet may struggle.")

        mean_intensity = arr.mean()
        if mean_intensity < 50:
            notes.append("Image appears very dark. Auto-invert will be applied.")

        return {
            "quality": quality,
            "stroke_coverage": round(coverage, 4),
            "mean_intensity": round(float(mean_intensity), 1),
            "notes": notes,
        }
