"""
ui/app.py
──────────
Gradio-based interactive web UI for Sketch-Guided Image Generation & Editing.

Tabs
────
1. 🎨 Sketch → Generate  — Draw a sketch + enter a text prompt → generate image
2. ✏️ Edit Image          — Upload a photo, draw edit strokes → inpaint the region
3. ⚡ Performance         — Latency dashboard and hardware info

Run:
    python ui/app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from PIL import Image

from src.pipeline import SketchPipeline
from src.optimizer import recommend_settings, get_vram_usage_gb
from utils.sketch_utils import canny_from_image, hed_from_image
from utils.image_utils import make_grid

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# Global pipeline (lazy-loaded on first inference)
# ─────────────────────────────────────────────────
_pipeline: SketchPipeline | None = None


def get_pipeline() -> SketchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SketchPipeline()
    return _pipeline


# ─────────────────────────────────────────────────
# Generation tab handler
# ─────────────────────────────────────────────────

def run_generation(
    sketch_data,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    num_images: int,
    seed_str: str,
    enhance_prompt: bool,
):
    """Called when user clicks 'Generate' on Tab 1."""
    if sketch_data is None:
        gr.Warning("Please draw a sketch first!")
        return None, "⚠️ No sketch provided."

    # Gradio Sketchpad / ImageEditor returns composite image
    if isinstance(sketch_data, dict):
        # New Gradio ≥4 ImageEditor format: {"background": ..., "layers": [...], "composite": ...}
        composite = sketch_data.get("composite") or sketch_data.get("background")
        sketch_img = Image.fromarray(composite).convert("RGB") if composite is not None else None
    else:
        sketch_img = Image.fromarray(sketch_data).convert("RGB") if sketch_data is not None else None

    if sketch_img is None:
        gr.Warning("Could not read sketch. Please try again.")
        return None, "⚠️ Sketch read error."

    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except ValueError:
            gr.Warning("Seed must be an integer. Using random seed.")

    try:
        pipe = get_pipeline()
        images, latency = pipe.generate(
            sketch=sketch_img,
            prompt=prompt or "a realistic image",
            negative_prompt=negative_prompt or None,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            num_images=num_images,
            seed=seed,
            enhance_prompt=enhance_prompt,
        )
        status = f"✅ Generated {len(images)} image(s) in **{latency:.2f}s**"
        if num_images == 1:
            return images[0], status
        else:
            return make_grid(images, cols=min(2, num_images)), status

    except Exception as exc:
        logger.exception("Generation failed")
        return None, f"❌ Error: {exc}"


# ─────────────────────────────────────────────────
# Editing tab handler
# ─────────────────────────────────────────────────

def run_editing(
    original_image,
    sketch_data,
    prompt: str,
    negative_prompt: str,
    mask_strategy: str,
    num_steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    inpaint_strength: float,
    num_images: int,
    seed_str: str,
):
    if original_image is None:
        gr.Warning("Please upload an image to edit!")
        return None, None, "⚠️ No image provided."

    if sketch_data is None:
        gr.Warning("Please draw edit strokes on the image!")
        return None, None, "⚠️ No strokes drawn."

    if isinstance(sketch_data, dict):
        composite = sketch_data.get("composite") or sketch_data.get("background")
        sketch_img = Image.fromarray(composite).convert("RGB") if composite is not None else None
    else:
        sketch_img = Image.fromarray(sketch_data).convert("RGB") if sketch_data is not None else None

    orig_img = Image.fromarray(original_image).convert("RGB") if not isinstance(original_image, Image.Image) else original_image

    seed = None
    if seed_str.strip():
        try:
            seed = int(seed_str.strip())
        except ValueError:
            pass

    try:
        pipe = get_pipeline()

        # Show mask preview
        mask_preview = pipe.preview_edit_mask(orig_img, sketch_img, strategy=mask_strategy)

        images, latency = pipe.edit(
            original=orig_img,
            sketch_overlay=sketch_img,
            prompt=prompt or "realistic, detailed",
            negative_prompt=negative_prompt or None,
            mask_strategy=mask_strategy,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            strength=inpaint_strength,
            num_images=num_images,
            seed=seed,
            blend_result=True,
        )

        status = f"✅ Edited in **{latency:.2f}s**"
        return images[0], mask_preview, status

    except Exception as exc:
        logger.exception("Editing failed")
        return None, None, f"❌ Error: {exc}"


# ─────────────────────────────────────────────────
# Edge extraction helper
# ─────────────────────────────────────────────────

def extract_edges(image, method: str) -> Image.Image | None:
    if image is None:
        return None
    img = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image
    if method == "Canny":
        return canny_from_image(img)
    elif method == "HED (soft edges)":
        return hed_from_image(img)
    return None


# ─────────────────────────────────────────────────
# Performance tab handler
# ─────────────────────────────────────────────────

def get_perf_stats():
    import torch
    lines = []

    device = "Unknown"
    if torch.cuda.is_available():
        device = f"CUDA — {torch.cuda.get_device_name(0)}"
        vram_used = get_vram_usage_gb()
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        lines.append(f"**GPU:** {device}")
        lines.append(f"**VRAM:** {vram_used:.1f} / {vram_total:.0f} GB used")
    elif torch.backends.mps.is_available():
        lines.append("**Device:** Apple MPS")
    else:
        lines.append("**Device:** CPU")

    if _pipeline:
        stats = _pipeline.latency_stats()
        gen = stats["generation"]
        edit = stats["editing"]
        if gen.get("n", 0) > 0:
            lines.append(f"\n**Generation latency:**")
            lines.append(f"- Mean: {gen['mean_s']}s | P95: {gen['p95_s']}s | Samples: {gen['n']}")
        if edit.get("n", 0) > 0:
            lines.append(f"\n**Editing latency:**")
            lines.append(f"- Mean: {edit['mean_s']}s | P95: {edit['p95_s']}s | Samples: {edit['n']}")

    return "\n".join(lines) if lines else "No data yet — run some inferences first."


# ─────────────────────────────────────────────────
# Build and launch Gradio UI
# ─────────────────────────────────────────────────

CSS = """
body { font-family: 'IBM Plex Mono', monospace; }
.tab-nav button { font-size: 1.05rem; }
#status_gen, #status_edit { font-size: 0.9rem; }
.gr-button-primary { background: #1a1a2e !important; }
footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(
        title="Sketch → Image  ✏️🎨",
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="cyan",
            neutral_hue="zinc",
        ),
        css=CSS,
    ) as demo:

        gr.Markdown(
            """
            # ✏️ Sketch-Guided Image Generation & Editing
            *Draw → Describe → Generate.  Near real-time with LCM-LoRA + ControlNet.*
            """
        )

        # ── Tab 1: Sketch → Generate ─────────────────
        with gr.Tab("🎨 Sketch → Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Draw your sketch")
                    sketch_canvas = gr.Sketchpad(
                        label="Sketch Canvas",
                        type="numpy",
                        height=512,
                        width=512,
                        brush=gr.Brush(colors=["#000000"], default_color="#000000", default_size=4),
                    )

                    gr.Markdown("### 2. Describe the image")
                    prompt_gen = gr.Textbox(
                        label="Prompt",
                        placeholder='e.g. "a golden retriever sitting on green grass, sunny day"',
                        lines=2,
                    )
                    neg_prompt_gen = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="e.g. blurry, low quality",
                        lines=1,
                    )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        num_steps_gen = gr.Slider(1, 30, value=4, step=1, label="Inference Steps")
                        guidance_gen = gr.Slider(0.5, 15.0, value=1.5, step=0.5, label="Guidance Scale")
                        cn_scale_gen = gr.Slider(0.0, 2.0, value=0.9, step=0.05, label="ControlNet Scale")
                        num_images_gen = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        seed_gen = gr.Textbox(label="Seed (leave blank for random)", value="")
                        enhance_prompt_gen = gr.Checkbox(label="Auto-enhance prompt", value=True)

                    btn_gen = gr.Button("🚀 Generate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 3. Result")
                    output_gen = gr.Image(label="Generated Image", height=512)
                    status_gen = gr.Markdown("", elem_id="status_gen")

            btn_gen.click(
                fn=run_generation,
                inputs=[
                    sketch_canvas, prompt_gen, neg_prompt_gen,
                    num_steps_gen, guidance_gen, cn_scale_gen,
                    num_images_gen, seed_gen, enhance_prompt_gen,
                ],
                outputs=[output_gen, status_gen],
            )

        # ── Tab 2: Edit Image ─────────────────────────
        with gr.Tab("✏️ Edit Image"):
            gr.Markdown(
                "Upload an image, draw strokes over the region you want to change, "
                "then describe what should appear there."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload your image")
                    upload_img = gr.Image(label="Original Image", height=400)

                    gr.Markdown("### 2. Draw edit strokes")
                    edit_canvas = gr.Sketchpad(
                        label="Draw strokes over the region to edit",
                        type="numpy",
                        height=400,
                        brush=gr.Brush(colors=["#FF4500"], default_color="#FF4500", default_size=6),
                    )

                    with gr.Row():
                        edge_method = gr.Dropdown(
                            ["Canny", "HED (soft edges)"], label="Extract edges from photo", value="Canny"
                        )
                        btn_edges = gr.Button("Extract Edges as Sketch")

                    gr.Markdown("### 3. Describe the edit")
                    prompt_edit = gr.Textbox(
                        label="What should appear in the edited region?",
                        placeholder='e.g. "a window with white frame", "tall oak tree"',
                        lines=2,
                    )
                    neg_prompt_edit = gr.Textbox(
                        label="Negative Prompt (optional)",
                        lines=1,
                    )

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        mask_strat = gr.Dropdown(
                            ["dilate", "flood_fill", "convex_hull", "bounding_box"],
                            value="dilate",
                            label="Mask Strategy",
                        )
                        num_steps_edit = gr.Slider(1, 30, value=4, step=1, label="Inference Steps")
                        guidance_edit = gr.Slider(0.5, 15.0, value=1.5, step=0.5, label="Guidance Scale")
                        cn_scale_edit = gr.Slider(0.0, 2.0, value=0.9, step=0.05, label="ControlNet Scale")
                        strength_edit = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Inpainting Strength")
                        num_images_edit = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        seed_edit = gr.Textbox(label="Seed", value="")

                    btn_edit = gr.Button("✏️ Apply Edit", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    output_edit = gr.Image(label="Edited Image", height=400)
                    mask_preview = gr.Image(label="Mask Preview (region to be edited)", height=200)
                    status_edit = gr.Markdown("", elem_id="status_edit")

            # Extract edges → populates the sketch canvas
            btn_edges.click(
                fn=extract_edges,
                inputs=[upload_img, edge_method],
                outputs=[edit_canvas],
            )

            btn_edit.click(
                fn=run_editing,
                inputs=[
                    upload_img, edit_canvas, prompt_edit, neg_prompt_edit,
                    mask_strat, num_steps_edit, guidance_edit, cn_scale_edit,
                    strength_edit, num_images_edit, seed_edit,
                ],
                outputs=[output_edit, mask_preview, status_edit],
            )

        # ── Tab 3: Performance ────────────────────────
        with gr.Tab("⚡ Performance"):
            gr.Markdown("## Hardware & Latency Dashboard")
            perf_md = gr.Markdown("Click 'Refresh' to load stats.")
            btn_perf = gr.Button("🔄 Refresh Stats")
            btn_perf.click(fn=get_perf_stats, outputs=perf_md)

            gr.Markdown(
                """
                ### Optimisation Tips
                | Technique | Speedup | Notes |
                |-----------|---------|-------|
                | LCM-LoRA (4 steps) | 5× | Default; best for GPU |
                | FP16 precision | 2× | Auto-enabled on CUDA |
                | xformers attention | 1.3× | Install `xformers` |
                | `torch.compile` | 1.3× | PyTorch 2.x; slow first run |
                | Reduce image size (384px) | 1.8× | Slight quality drop |
                | VAE tiling | memory | Needed for <6 GB VRAM |
                """
            )

        # ── Footer ────────────────────────────────────
        gr.Markdown(
            """
            ---
            *Powered by [ControlNet](https://github.com/lllyasviel/ControlNet) + 
            [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model) + 
            [Stable Diffusion](https://github.com/CompVis/stable-diffusion)*
            """
        )

    return demo


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ui_cfg = cfg.get("ui", {})

    demo = build_ui()
    demo.launch(
        server_name=ui_cfg.get("host", "0.0.0.0"),
        server_port=ui_cfg.get("port", 7860),
        share=ui_cfg.get("share", False),
        show_error=True,
    )
