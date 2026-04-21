"""
utils/image_utils.py
─────────────────────
General-purpose image helpers shared across the pipeline.
"""

from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from disk as PIL RGB."""
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: Union[str, Path], quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower().lstrip(".")
    fmt_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}
    img.save(path, format=fmt_map.get(fmt, "PNG"), quality=quality)


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL image to base64 string (for embedding in HTML/JSON)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


# ──────────────────────────────────────────────
# Resizing / padding
# ──────────────────────────────────────────────

def resize_to_multiple(
    img: Image.Image,
    multiple: int = 8,
    max_size: int = 1024,
) -> Image.Image:
    """
    Resize image so both dimensions are multiples of `multiple` and
    neither exceeds `max_size`.  Preserves aspect ratio.
    """
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)
    nw = max(multiple, round(w * scale / multiple) * multiple)
    nh = max(multiple, round(h * scale / multiple) * multiple)
    return img.resize((nw, nh), Image.LANCZOS)


def pad_to_square(img: Image.Image, fill: int = 255) -> Image.Image:
    """Pad image with `fill` colour so it becomes square."""
    w, h = img.size
    side = max(w, h)
    out = Image.new(img.mode, (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Crop the centre of an image to `size` (w, h)."""
    w, h = img.size
    tw, th = size
    left = (w - tw) // 2
    top = (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


# ──────────────────────────────────────────────
# Blending / compositing
# ──────────────────────────────────────────────

def blend_images(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
    blur_mask: int = 8,
) -> Image.Image:
    """
    Composites `edited` onto `original` using `mask` (white = edited region).
    Feathers the mask edges with a Gaussian blur for seamless blending.

    Parameters
    ----------
    original  : Source image (background).
    edited    : New content to paste into the masked region.
    mask      : Greyscale (L) mask; white = use edited, black = keep original.
    blur_mask : Gaussian blur radius for feathering; 0 = hard edges.
    """
    original = original.convert("RGBA")
    edited = edited.convert("RGBA").resize(original.size, Image.LANCZOS)
    mask = mask.convert("L").resize(original.size, Image.LANCZOS)

    if blur_mask > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_mask))

    result = Image.composite(edited, original, mask)
    return result.convert("RGB")


def overlay_sketch_on_image(
    image: Image.Image,
    sketch: Image.Image,
    alpha: float = 0.4,
    stroke_colour: Tuple[int, int, int] = (255, 80, 0),
) -> Image.Image:
    """
    Overlay coloured sketch strokes on an image (for debug / preview).
    `sketch` should be L-mode with black strokes on white background.
    """
    image = image.convert("RGBA")
    sketch_l = sketch.convert("L").resize(image.size, Image.LANCZOS)
    stroke_arr = np.array(sketch_l)

    # Where sketch is dark → draw strokes
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_mask = stroke_arr < 128      # True where strokes exist
    r, g, b = stroke_colour
    for y in range(stroke_mask.shape[0]):
        for x in range(stroke_mask.shape[1]):
            if stroke_mask[y, x]:
                overlay.putpixel((x, y), (r, g, b, int(alpha * 255)))

    return Image.alpha_composite(image, overlay).convert("RGB")


# ──────────────────────────────────────────────
# Quality / diagnostics helpers
# ──────────────────────────────────────────────

def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute SSIM between two images (requires scikit-image)."""
    try:
        from skimage.metrics import structural_similarity as ssim
        a = np.array(img_a.convert("L"))
        b = np.array(img_b.convert("L").resize(img_a.size, Image.LANCZOS))
        return float(ssim(a, b, data_range=255))
    except ImportError:
        return -1.0


def make_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    """Arrange a list of PIL images into a grid."""
    if not images:
        return Image.new("RGB", (512, 512), 255)
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), 255)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img.resize((w, h), Image.LANCZOS), (c * w, r * h))
    return grid
