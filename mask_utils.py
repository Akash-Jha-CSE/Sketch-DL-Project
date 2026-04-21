"""
utils/mask_utils.py
────────────────────
Automatically derive inpainting masks from user-drawn sketch strokes.

When a user draws strokes on an existing image to indicate *where* they want
an edit, we need to convert those strokes into a binary mask that covers the
intended edit region (not just the stroke pixels themselves).

Strategies
──────────
1. Dilate strokes into a broad mask (simplest).
2. Flood-fill the enclosed region of a closed sketch shape.
3. Use convex-hull of all stroke pixels.
4. Bounding-box of all stroke pixels (fast, coarse).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def strokes_to_mask(
    sketch: Image.Image,
    strategy: str = "dilate",
    *,
    dilation: int = 20,
    blur: int = 8,
    threshold: int = 128,
) -> Image.Image:
    """
    Convert a sketch overlay (black strokes on white background) into a
    binary inpainting mask (white = region to repaint).

    Parameters
    ----------
    sketch    : PIL L or RGB image; dark pixels = strokes.
    strategy  : "dilate" | "flood_fill" | "convex_hull" | "bounding_box"
    dilation  : Dilation kernel radius (pixels) for the "dilate" strategy.
    blur      : Gaussian blur to soften mask edges (0 = hard edges).
    threshold : Pixel threshold for binarising the sketch.

    Returns
    -------
    PIL Image (L mode, 0–255); white = edit region, black = keep unchanged.
    """
    arr = np.array(sketch.convert("L"))
    # Invert: we want stroke pixels = 255 (foreground)
    binary = (arr < threshold).astype(np.uint8) * 255

    if strategy == "dilate":
        mask = _dilate_strategy(binary, dilation)
    elif strategy == "flood_fill":
        mask = _flood_fill_strategy(binary)
    elif strategy == "convex_hull":
        mask = _convex_hull_strategy(binary)
    elif strategy == "bounding_box":
        mask = _bounding_box_strategy(binary)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy!r}")

    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)

    return Image.fromarray(mask.clip(0, 255).astype(np.uint8), mode="L")


# ──────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────

def _dilate_strategy(binary: np.ndarray, radius: int) -> np.ndarray:
    """Expand strokes outward by `radius` pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel, iterations=1)


def _flood_fill_strategy(binary: np.ndarray) -> np.ndarray:
    """
    If the strokes form a roughly closed shape, flood-fill the interior.
    Falls back to dilation if no enclosed region is found.
    """
    h, w = binary.shape
    # Dilate a bit to close small gaps in the outline
    closed = cv2.dilate(binary, np.ones((5, 5), np.uint8))
    # Flood-fill from the four corners (exterior)
    mask_ext = np.zeros((h + 2, w + 2), np.uint8)
    filled = closed.copy()
    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(filled, mask_ext, (seed[1], seed[0]), 128)
    # Interior = pixels NOT reached by flood fill that are also not strokes
    interior = ((filled != 128) & (binary == 0)).astype(np.uint8) * 255
    # Combine interior + original strokes
    combined = np.clip(interior + binary, 0, 255).astype(np.uint8)
    if combined.sum() == 0:                  # fallback
        return _dilate_strategy(binary, 20)
    return combined


def _convex_hull_strategy(binary: np.ndarray) -> np.ndarray:
    """Draw the convex hull of all stroke pixels as the mask."""
    points = np.column_stack(np.where(binary > 0))  # (N, 2) → (row, col)
    if len(points) < 3:
        return _dilate_strategy(binary, 20)
    hull = cv2.convexHull(points[:, ::-1])  # needs (col, row)
    mask = np.zeros_like(binary)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _bounding_box_strategy(binary: np.ndarray) -> np.ndarray:
    """Fill the axis-aligned bounding box of all stroke pixels."""
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return binary
    mask = np.zeros_like(binary)
    mask[ys.min():ys.max(), xs.min():xs.max()] = 255
    return mask


# ──────────────────────────────────────────────
# Mask refinement helpers
# ──────────────────────────────────────────────

def expand_mask(
    mask: Image.Image,
    expand_pixels: int = 10,
    feather: int = 5,
) -> Image.Image:
    """
    Expand and feather an existing mask.
    Useful for ensuring that the edited region covers the full target area.
    """
    arr = np.array(mask.convert("L"))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1))
    expanded = cv2.dilate(arr, kernel, iterations=1)
    if feather > 0:
        expanded = cv2.GaussianBlur(expanded, (0, 0), feather)
    return Image.fromarray(expanded.clip(0, 255).astype(np.uint8), mode="L")


def invert_mask(mask: Image.Image) -> Image.Image:
    """Invert a binary mask."""
    return Image.fromarray(255 - np.array(mask.convert("L")), mode="L")


def combine_masks(
    mask_a: Image.Image,
    mask_b: Image.Image,
    mode: str = "union",
) -> Image.Image:
    """
    Combine two masks.
    mode: "union" | "intersection" | "difference"
    """
    a = np.array(mask_a.convert("L")).astype(float)
    b = np.array(mask_b.convert("L").resize(mask_a.size)).astype(float)
    if mode == "union":
        out = np.clip(a + b, 0, 255)
    elif mode == "intersection":
        out = np.minimum(a, b)
    elif mode == "difference":
        out = np.clip(a - b, 0, 255)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return Image.fromarray(out.astype(np.uint8), mode="L")


def visualise_mask(
    image: Image.Image,
    mask: Image.Image,
    colour: Tuple[int, int, int] = (255, 100, 100),
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay a coloured transparent mask on an image for preview.
    Useful in the UI to show the user which region will be edited.
    """
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_l = mask.convert("L").resize(base.size, Image.LANCZOS)
    mask_arr = np.array(mask_l)

    r, g, b = colour
    for y in range(mask_arr.shape[0]):
        for x in range(mask_arr.shape[1]):
            v = mask_arr[y, x]
            if v > 10:
                a_val = int(v / 255.0 * alpha * 255)
                overlay.putpixel((x, y), (r, g, b, a_val))

    return Image.alpha_composite(base, overlay).convert("RGB")
