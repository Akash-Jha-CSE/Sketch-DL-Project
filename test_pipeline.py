"""
tests/test_pipeline.py
───────────────────────
Unit + integration tests for the sketch-guided image generation system.

Tests are split into:
  - Unit tests (no model weights needed — mock the pipeline)
  - Integration tests (require actual model weights — skipped by default)

Run:
    pytest tests/test_pipeline.py -v                    # unit tests only
    pytest tests/test_pipeline.py -v --run-integration  # include heavy tests
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_sketch(size: int = 64) -> Image.Image:
    """Minimal synthetic sketch: black circle on white background."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = size // 4
    draw.ellipse([(m, m), (3 * m, 3 * m)], outline="black", width=2)
    return img


def make_photo(size: int = 64) -> Image.Image:
    """Synthetic photo: solid colour with a gradient."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(50, 200, size, dtype=np.uint8)
    arr[:, :, 1] = 100
    arr[:, :, 2] = 180
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# utils/sketch_utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSketchUtils:
    def test_preprocess_sketch_output_size(self):
        from utils.sketch_utils import preprocess_sketch
        sketch = make_sketch(128)
        result = preprocess_sketch(sketch, target_size=(64, 64))
        assert result.size == (64, 64)
        assert result.mode == "L"

    def test_preprocess_sketch_auto_invert_dark(self):
        """Dark background should be inverted to white background."""
        from utils.sketch_utils import preprocess_sketch
        # Create white strokes on black background
        dark = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(dark)
        draw.ellipse([(10, 10), (50, 50)], outline=255, width=3)
        result = preprocess_sketch(dark, target_size=(64, 64), auto_invert=True)
        arr = np.array(result)
        # After invert, background should be mostly white (>200)
        assert arr.mean() > 150

    def test_preprocess_sketch_no_invert(self):
        """White background sketch should stay white when auto_invert is off."""
        from utils.sketch_utils import preprocess_sketch
        sketch = make_sketch(64)
        result = preprocess_sketch(sketch, auto_invert=False)
        arr = np.array(result)
        assert arr.mean() > 150

    def test_canny_from_image_returns_correct_size(self):
        from utils.sketch_utils import canny_from_image
        photo = make_photo(128)
        result = canny_from_image(photo, target_size=(64, 64))
        assert result.size == (64, 64)
        assert result.mode == "L"

    def test_hed_from_image_returns_correct_size(self):
        from utils.sketch_utils import hed_from_image
        photo = make_photo(128)
        result = hed_from_image(photo, target_size=(64, 64))
        assert result.size == (64, 64)

    def test_augment_sketch_preserves_size(self):
        from utils.sketch_utils import augment_sketch
        sketch = make_sketch(64).convert("L")
        result = augment_sketch(sketch, flip_horizontal=True, elastic_distortion=True)
        assert result.size == sketch.size

    def test_sketch_to_controlnet_input_rgb(self):
        from utils.sketch_utils import sketch_to_controlnet_input
        sketch = make_sketch(64)
        result = sketch_to_controlnet_input(sketch, target_size=(64, 64))
        assert result.mode == "RGB"
        assert result.size == (64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# utils/mask_utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskUtils:
    def _make_stroke_sketch(self, size=64) -> Image.Image:
        """Sketch with a simple rectangle of strokes."""
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(15, 15), (48, 48)], outline=0, width=3)
        return img

    def test_dilate_strategy(self):
        from utils.mask_utils import strokes_to_mask
        sketch = self._make_stroke_sketch()
        mask = strokes_to_mask(sketch, strategy="dilate", dilation=5, blur=0)
        arr = np.array(mask)
        # Mask should have some white pixels
        assert arr.max() == 255
        assert arr.mean() > 0

    def test_convex_hull_strategy(self):
        from utils.mask_utils import strokes_to_mask
        sketch = self._make_stroke_sketch()
        mask = strokes_to_mask(sketch, strategy="convex_hull", blur=0)
        arr = np.array(mask)
        assert arr.max() == 255

    def test_bounding_box_strategy(self):
        from utils.mask_utils import strokes_to_mask
        sketch = self._make_stroke_sketch(64)
        mask = strokes_to_mask(sketch, strategy="bounding_box", blur=0)
        arr = np.array(mask)
        assert arr.max() == 255

    def test_flood_fill_strategy(self):
        from utils.mask_utils import strokes_to_mask
        sketch = self._make_stroke_sketch()
        mask = strokes_to_mask(sketch, strategy="flood_fill", blur=0)
        arr = np.array(mask)
        assert arr.max() == 255

    def test_expand_mask(self):
        from utils.mask_utils import expand_mask
        small_mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(small_mask).ellipse([(20, 20), (40, 40)], fill=255)
        expanded = expand_mask(small_mask, expand_pixels=5, feather=0)
        # Expanded mask should have more white pixels
        assert np.array(expanded).sum() >= np.array(small_mask).sum()

    def test_combine_masks_union(self):
        from utils.mask_utils import combine_masks
        a = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(a).rectangle([(0, 0), (31, 63)], fill=255)
        b = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(b).rectangle([(32, 0), (63, 63)], fill=255)
        union = combine_masks(a, b, mode="union")
        arr = np.array(union)
        # Union should be nearly all white
        assert arr.mean() > 200

    def test_invert_mask(self):
        from utils.mask_utils import invert_mask
        mask = Image.new("L", (64, 64), 255)
        inverted = invert_mask(mask)
        assert np.array(inverted).mean() == 0


# ─────────────────────────────────────────────────────────────────────────────
# utils/image_utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_resize_to_multiple(self):
        from utils.image_utils import resize_to_multiple
        img = Image.new("RGB", (100, 75))
        resized = resize_to_multiple(img, multiple=8)
        w, h = resized.size
        assert w % 8 == 0
        assert h % 8 == 0

    def test_pad_to_square(self):
        from utils.image_utils import pad_to_square
        img = Image.new("RGB", (100, 60))
        squared = pad_to_square(img)
        assert squared.size[0] == squared.size[1] == 100

    def test_center_crop(self):
        from utils.image_utils import center_crop
        img = Image.new("RGB", (100, 100))
        cropped = center_crop(img, (64, 64))
        assert cropped.size == (64, 64)

    def test_make_grid(self):
        from utils.image_utils import make_grid
        images = [Image.new("RGB", (64, 64), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
        grid = make_grid(images, cols=2)
        assert grid.size == (128, 128)

    def test_pil_base64_roundtrip(self):
        from utils.image_utils import pil_to_base64, base64_to_pil
        img = make_photo(64)
        b64 = pil_to_base64(img)
        recovered = base64_to_pil(b64)
        assert recovered.size == img.size

    def test_blend_images(self):
        from utils.image_utils import blend_images
        bg = Image.new("RGB", (64, 64), (200, 200, 200))
        fg = Image.new("RGB", (64, 64), (100, 100, 100))
        mask = Image.new("L", (64, 64), 255)     # fully white = use fg
        result = blend_images(bg, fg, mask, blur_mask=0)
        arr = np.array(result)
        # Should be close to fg colour
        assert arr.mean() < 150


# ─────────────────────────────────────────────────────────────────────────────
# src/controlnet_wrapper tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestControlNetWrapper:
    def _make_wrapper(self):
        from src.controlnet_wrapper import ControlNetWrapper
        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        dummy_result = MagicMock()
        dummy_result.images = [Image.new("RGB", (64, 64), "blue")]
        mock_pipe.return_value = dummy_result
        cfg = {
            "inference": {
                "num_inference_steps": 4,
                "guidance_scale": 1.5,
                "controlnet_conditioning_scale": 0.9,
                "image_width": 64,
                "image_height": 64,
                "default_negative_prompt": "blurry",
            }
        }
        return ControlNetWrapper(mock_pipe, cfg)

    def test_enhance_prompt_appends_boosters(self):
        from src.controlnet_wrapper import ControlNetWrapper
        enhanced = ControlNetWrapper.enhance_prompt("a cat")
        assert "highly detailed" in enhanced
        assert "a cat" in enhanced

    def test_make_negative_prompt(self):
        from src.controlnet_wrapper import ControlNetWrapper
        neg = ControlNetWrapper.make_negative_prompt("extra limbs")
        assert "extra limbs" in neg
        assert "blurry" in neg

    def test_generate_calls_pipe(self):
        wrapper = self._make_wrapper()
        sketch = Image.new("RGB", (64, 64), "white")
        images, latency = wrapper.generate("a cat", sketch)
        assert len(images) == 1
        assert latency >= 0


# ─────────────────────────────────────────────────────────────────────────────
# src/optimizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizer:
    def test_latency_tracker(self):
        from src.optimizer import LatencyTracker
        tracker = LatencyTracker(window=5)
        for t in [0.5, 0.6, 0.4, 0.7, 0.5]:
            tracker.record(t)
        assert abs(tracker.mean - 0.54) < 0.01
        summary = tracker.summary()
        assert summary["n"] == 5
        assert "mean_s" in summary

    def test_recommend_settings_cpu(self):
        import torch
        from src.optimizer import recommend_settings
        rec = recommend_settings(torch.device("cpu"))
        assert rec["use_fp16"] is False
        assert "note" in rec

    def test_free_vram_no_crash(self):
        from src.optimizer import free_vram
        free_vram()  # Should not raise even without GPU


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (skipped unless --run-integration flag is set)
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False)


@pytest.fixture
def run_integration(request):
    return request.config.getoption("--run-integration")


@pytest.mark.slow
class TestIntegration:
    """
    End-to-end tests that require downloaded model weights.
    Run with:  pytest tests/test_pipeline.py -v --run-integration
    """

    def test_generate_from_sketch(self, run_integration):
        if not run_integration:
            pytest.skip("Integration tests disabled. Use --run-integration.")

        from src.pipeline import SketchPipeline
        pipe = SketchPipeline()
        sketch = make_sketch(512)
        images, latency = pipe.generate(
            sketch=sketch,
            prompt="a golden retriever on grass",
            num_steps=4,
            num_images=1,
        )
        assert len(images) == 1
        assert images[0].size == (512, 512)
        assert latency < 30, f"Generation too slow: {latency:.1f}s"

    def test_edit_image(self, run_integration):
        if not run_integration:
            pytest.skip("Integration tests disabled. Use --run-integration.")

        from src.pipeline import SketchPipeline
        pipe = SketchPipeline()
        photo = make_photo(512)
        strokes = make_sketch(512)
        images, latency = pipe.edit(
            original=photo,
            sketch_overlay=strokes,
            prompt="a window with white frame",
            num_steps=4,
        )
        assert len(images) == 1
        assert latency < 30


# ─────────────────────────────────────────────────────────────────────────────
# Run with pytest
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
