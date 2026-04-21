# Sketch-Guided Real-Time Image Generation & Editing

A deep learning system for converting hand-drawn sketches (optionally with text prompts) into
realistic images, and for sketch-guided inpainting/editing of existing images — with near
real-time inference.

---

## Architecture Overview

```
sketch_guided_imggen/
├── src/
│   ├── pipeline.py          # Core generation & editing pipeline
│   ├── sketch_processor.py  # Sketch pre-processing (denoising, edge enhancement)
│   ├── controlnet_wrapper.py# ControlNet conditioning wrapper
│   ├── inpainting.py        # Sketch-guided inpainting / editing
│   ├── optimizer.py         # Inference optimizations (quantization, ONNX, TensorRT)
│   └── latent_consistency.py# LCM (Latent Consistency Models) for fast inference
├── models/
│   └── model_manager.py     # Lazy model loading, caching, memory management
├── utils/
│   ├── image_utils.py       # Image I/O, resizing, blending helpers
│   ├── sketch_utils.py      # Sketch augmentation, Canny/HED edge detection
│   └── mask_utils.py        # Mask generation from sketch strokes
├── ui/
│   ├── app.py               # Gradio web application (main entry point)
│   └── components.py        # Reusable UI components
├── configs/
│   └── config.yaml          # All hyper-parameters and model paths
├── scripts/
│   ├── download_models.py   # One-shot model downloader
│   └── benchmark.py         # Latency / quality benchmarking
├── tests/
│   └── test_pipeline.py     # Unit + integration tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

```bash
python scripts/download_models.py
```

This downloads:
- **Stable Diffusion v1.5** (base generator)
- **ControlNet Scribble** (sketch conditioning)
- **LCM-LoRA** (4–8 step fast inference)
- **Segment Anything Model** (for mask generation)

### 3. Launch the web UI

```bash
python ui/app.py
```

Open `http://localhost:7860` in your browser.

---

## Features

| Feature | Description |
|---|---|
| Sketch → Image | Draw a sketch + optional text prompt → realistic image |
| Sketch-guided editing | Upload an image, draw strokes → edit that region only |
| Fast inference | LCM / SDXL-Turbo path for ~0.3–1 s on GPU |
| Batch generation | Generate N variations from the same sketch |
| Sketch preprocessing | Auto-denoise, edge-sharpen, threshold noisy freehand input |
| Mask auto-generation | Derive inpainting masks automatically from stroke regions |

---

## Inference Speed Guide

| Mode | Steps | GPU (A10) | CPU |
|---|---|---|---|
| LCM (fast) | 4 | ~0.3 s | ~8 s |
| Standard SD | 20 | ~1.2 s | ~60 s |
| SDXL-Turbo | 1–4 | ~0.5 s | N/A |

---

## Configuration (`configs/config.yaml`)

Key parameters you can tune:

```yaml
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  controlnet: "lllyasviel/control_v11p_sd15_scribble"
  use_lcm: true
  lcm_lora: "latent-consistency/lcm-lora-sdv1-5"

inference:
  num_inference_steps: 4       # 4 for LCM, 20 for standard
  guidance_scale: 1.5          # Low for LCM, 7.5 for standard
  controlnet_scale: 0.9        # How strongly sketch guides generation
  image_size: 512

optimization:
  use_fp16: true
  enable_xformers: true
  compile_model: false          # torch.compile (requires PyTorch 2.x)
  use_onnx: false
```
