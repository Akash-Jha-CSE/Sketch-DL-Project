"""
utils/sketch_utils.py
─────────────────────
Preprocessing utilities that turn noisy freehand sketches into clean edge maps
that ControlNet can reliably condition on.

Key operations
──────────────
1. Threshold / binarise                  → remove grey noise
2. Gaussian denoise                      → smooth jagged strokes
3. Morphological clean-up               → close small gaps, remove speckles
4. Canny edge detection                 → alternative when input is a photo
5. HED-style soft edge estimation      → richer gradient edges from photos
6. Invert / normalise                   → ensure black strokes on white bg
7. Augmentation helpers                 → random distortion for training data
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Tuple, Optional


# ──────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL RGBA/RGB/L → BGR numpy array for OpenCV."""
    img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array → PIL RGB image."""
    if arr.ndim == 2:                        # grayscale
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────
# Main sketch processing pipeline
# ──────────────────────────────────────────────

def preprocess_sketch(
    sketch: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    *,
    denoise_sigma: float = 1.0,
    threshold: int = 128,
    auto_invert: bool = True,
    dilate_strokes: int = 0,
    close_gaps: int = 0,
) -> Image.Image:
    """
    Full preprocessing pipeline for a freehand sketch.

    Parameters
    ----------
    sketch        : Input PIL image (any mode).
    target_size   : (width, height) to resize output to.
    denoise_sigma : Gaussian blur sigma to reduce noise; 0 = off.
    threshold     : Binarisation threshold (0–255).
    auto_invert   : Invert if the image looks like white-on-black.
    dilate_strokes: Morphological dilation kernel size (makes lines thicker).
    close_gaps    : Morphological closing kernel size (joins nearby strokes).

    Returns
    -------
    PIL Image (L mode) — black strokes on white background, sized target_size.
    """
    # 1. Convert to greyscale
    grey = sketch.convert("L")

    # 2. Resize to target
    grey = grey.resize(target_size, Image.LANCZOS)
    arr: np.ndarray = np.array(grey, dtype=np.uint8)

    # 3. Denoise
    if denoise_sigma > 0:
        ksize = int(6 * denoise_sigma + 1)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        arr = cv2.GaussianBlur(arr, (ksize, ksize), denoise_sigma)

    # 4. Auto-invert detection: if most pixels are dark, treat as white-on-black
    if auto_invert:
        mean_val = arr.mean()
        if mean_val < 127:           # dark background → invert
            arr = 255 - arr

    # 5. Binarise
    _, arr = cv2.threshold(arr, threshold, 255, cv2.THRESH_BINARY)

    # 6. Morphological operations
    if close_gaps > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_gaps, close_gaps))
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

    if dilate_strokes > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_strokes, dilate_strokes))
        arr = cv2.dilate(arr, kernel, iterations=1)

    # 7. Remove tiny speckles (noise)
    arr = _remove_small_components(arr, min_area=50)

    return Image.fromarray(arr, mode="L")


def _remove_small_components(binary: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Remove connected components smaller than `min_area` pixels."""
    # Invert for connectedComponents (expects white blobs on black)
    inv = 255 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    clean = np.zeros_like(inv)
    for label in range(1, num_labels):                       # skip background (0)
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return 255 - clean


# ──────────────────────────────────────────────
# Edge extraction from photographs
# ──────────────────────────────────────────────

def canny_from_image(
    image: Image.Image,
    low: int = 50,
    high: int = 150,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Extract Canny edges from a photograph to use as a sketch proxy.
    Returns black strokes on white background.
    """
    arr = pil_to_cv(image.resize(target_size, Image.LANCZOS))
    grey = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    # Invert: Canny returns white edges on black; ControlNet wants black on white
    return Image.fromarray(255 - edges, mode="L")


def hed_from_image(image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Lightweight HED-style edge map using multi-scale Laplacian of Gaussian.
    Full HED requires a pretrained model; this is a fast approximation that
    produces softer, richer edges than plain Canny — suitable for ControlNet.
    """
    arr = pil_to_cv(image.resize(target_size, Image.LANCZOS))
    grey = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    edges = np.zeros_like(grey)
    for sigma in [1, 2, 4]:
        blurred = cv2.GaussianBlur(grey, (0, 0), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F)
        edges += np.abs(lap)

    # Normalise to 0–255
    edges = (edges / edges.max() * 255).clip(0, 255).astype(np.uint8)
    # Invert: dark strokes on white background
    return Image.fromarray(255 - edges, mode="L")


# ──────────────────────────────────────────────
# Training-time augmentation helpers
# ──────────────────────────────────────────────

def augment_sketch(
    sketch: Image.Image,
    *,
    max_rotation: float = 10.0,
    max_translate: float = 0.05,
    flip_horizontal: bool = True,
    elastic_distortion: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """
    Apply random augmentations to a sketch for training data diversity.
    All augmentations are structure-preserving (no colour changes).
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = np.array(sketch)
    h, w = arr.shape[:2]

    # Horizontal flip
    if flip_horizontal and rng.random() < 0.5:
        arr = np.fliplr(arr)

    # Random rotation + translation via affine transform
    angle = rng.uniform(-max_rotation, max_rotation)
    tx = rng.uniform(-max_translate, max_translate) * w
    ty = rng.uniform(-max_translate, max_translate) * h
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Elastic distortion (simulates natural hand-drawing variance)
    if elastic_distortion:
        arr = _elastic_distort(arr, alpha=30, sigma=5, rng=rng)

    return Image.fromarray(arr)


def _elastic_distort(
    arr: np.ndarray,
    alpha: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply elastic deformation as in Simard et al. (2003)."""
    h, w = arr.shape[:2]
    dx = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32) * 2 - 1,
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32) * 2 - 1,
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# ──────────────────────────────────────────────
# Utility: convert sketch canvas to ControlNet input
# ──────────────────────────────────────────────

def sketch_to_controlnet_input(
    sketch: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    cfg: Optional[dict] = None,
) -> Image.Image:
    """
    High-level helper used by the pipeline.
    Applies full preprocessing and returns an RGB image (black lines, white bg)
    ready to pass to ControlNet as the conditioning image.
    """
    cfg = cfg or {}
    processed = preprocess_sketch(
        sketch,
        target_size=target_size,
        denoise_sigma=cfg.get("denoise_sigma", 1.0),
        threshold=cfg.get("threshold_value", 128),
        auto_invert=cfg.get("auto_invert", True),
        dilate_strokes=cfg.get("dilate_strokes", 0),
        close_gaps=cfg.get("close_gaps", 0),
    )
    # ControlNet scribble expects RGB
    return processed.convert("RGB")
