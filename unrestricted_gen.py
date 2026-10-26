#!/usr/bin/env python3
"""
Unrestricted Text-to-Image + Edit (SDXL / SD1.5)
Safety checker disabled. Adult content allowed. No minors.
Requires a CUDA GPU (e.g. Colab T4 or local NVIDIA).
"""

from __future__ import annotations

import argparse
import gc
import os

import gradio as gr
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Models (all run with safety filters off)
# ---------------------------------------------------------------------------
MODELS = {
    "SDXL Base (general)": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "kind": "sdxl",
    },
    "Pony Realism SDXL (NSFW-friendly)": {
        "id": "John6666/pony-realism-v21main-sdxl",
        "kind": "sdxl",
    },
    "SD 1.5 (classic, lighter VRAM)": {
        "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "kind": "sd15",
    },
}

DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, deformed, ugly, bad anatomy, "
    "extra limbs, missing fingers, mutated hands, poorly drawn face, "
    "watermark, text, signature, child, underage, loli, young"
)

# Global state
_state: dict = {"pipe": None, "pipe_i2i": None, "model_key": None, "kind": None}


def _clear_gpu() -> None:
    _state["pipe"] = None
    _state["pipe_i2i"] = None
    _state["model_key"] = None
    _state["kind"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_key: str, status=gr.Progress(track_tqdm=True)):
    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU required. In Colab: Runtime → Change runtime type → T4 GPU.")

    if model_key not in MODELS:
        raise gr.Error(f"Unknown model: {model_key}")

    if _state["model_key"] == model_key and _state["pipe"] is not None:
        return f"Already loaded: {model_key}"

    status(0, desc=f"Loading {model_key}…")
    _clear_gpu()

    info = MODELS[model_key]
    model_id = info["id"]
    kind = info["kind"]
    dtype = torch.float16

    if kind == "sdxl":
        # SDXL has no built-in safety checker; load once and share weights for img2img
        load_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=True,
            add_watermarker=False,
        )
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, variant="fp16", **load_kwargs
            )
        except Exception:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)
        pipe = pipe.to("cuda")
        pipe_i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    else:
        # SD 1.5: explicitly disable the NSFW safety checker
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")
        pipe_i2i = StableDiffusionImg2ImgPipeline(**pipe.components)

    pipe.enable_attention_slicing()
    pipe_i2i.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
        pipe_i2i.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
        pipe_i2i.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _state["pipe"] = pipe
    _state["pipe_i2i"] = pipe_i2i
    _state["model_key"] = model_key
    _state["kind"] = kind

    vram = torch.cuda.memory_allocated() / 1024**3
    return f"Loaded {model_key} ({vram:.1f} GB VRAM in use)"


def _generator(seed: int):
    if seed is None or int(seed) < 0:
        return None
    return torch.Generator(device="cuda").manual_seed(int(seed))


def _ensure_loaded():
    if _state["pipe"] is None:
        raise gr.Error("Load a model first (Load Model button).")


def generate_image(
    prompt: str,
    negative_prompt: str,
    steps: float,
    guidance: float,
    width: float,
    height: float,
    seed: float,
    num_images: float,
):
    _ensure_loaded()
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt.")

    kind = _state["kind"]
    # SD1.5 prefers multiples of 8 around 512; SDXL around 1024
    w, h = int(width), int(height)
    if kind == "sd15":
        w = max(384, min(768, w - (w % 8)))
        h = max(384, min(768, h - (h % 8)))
    else:
        w = max(768, min(1280, w - (w % 64)))
        h = max(768, min(1280, h - (h % 64)))

    n = max(1, min(4, int(num_images)))
    images = []
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else None

    for i in range(n):
        s = (base_seed + i) if base_seed is not None else -1
        gen = _generator(s)
        out = _state["pipe"](
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=w,
            height=h,
            generator=gen,
        )
        images.append(out.images[0])

    return images


def edit_image(
    init_image: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: float,
    guidance: float,
    seed: float,
):
    _ensure_loaded()
    if init_image is None:
        raise gr.Error("Upload an image to edit.")
    if not (prompt or "").strip():
        raise gr.Error("Enter an edit prompt.")

    init_image = init_image.convert("RGB")
    out = _state["pipe_i2i"](
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        image=init_image,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=_generator(int(seed) if seed is not None else -1),
    )
    return out.images[0]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Unrestricted Image Generator") as demo:
        gr.Markdown(
            """
# Unrestricted Image Generator
Text-to-image + img2img. **Safety checker OFF.** Adult content allowed.

**Rules:** adults only — no minors / underage content. Use a GPU (Colab T4 or local CUDA).
            """
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model",
            )
            load_btn = gr.Button("Load Model", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, value="No model loaded")
        load_btn.click(load_model, inputs=[model_dd], outputs=[status])

        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Detailed prompt… (subject, style, lighting, camera)",
                )
                neg = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE, lines=2)
                with gr.Row():
                    steps = gr.Slider(15, 50, value=28, step=1, label="Steps")
                    guidance = gr.Slider(1, 15, value=7, step=0.5, label="Guidance")
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
                with gr.Row():
                    width = gr.Slider(512, 1280, value=1024, step=64, label="Width")
                    height = gr.Slider(512, 1280, value=1024, step=64, label="Height")
                    seed = gr.Number(value=-1, label="Seed (−1 = random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
                out = gr.Gallery(label="Result", columns=2, height=512)
                btn.click(
                    generate_image,
                    [prompt, neg, steps, guidance, width, height, seed, num_images],
                    out,
                )

            with gr.Tab("Edit Image"):
                init = gr.Image(type="pil", label="Upload image", height=320)
                prompt_e = gr.Textbox(label="Edit Prompt", lines=3)
                neg_e = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE, lines=2)
                with gr.Row():
                    strength = gr.Slider(0.2, 0.95, value=0.55, step=0.05, label="Strength")
                    steps_e = gr.Slider(15, 50, value=28, step=1, label="Steps")
                    guidance_e = gr.Slider(1, 15, value=7, step=0.5, label="Guidance")
                seed_e = gr.Number(value=-1, label="Seed (−1 = random)", precision=0)
                btn_e = gr.Button("Edit", variant="primary")
                out_e = gr.Image(label="Edited Result", height=512)
                btn_e.click(
                    edit_image,
                    [init, prompt_e, neg_e, strength, steps_e, guidance_e, seed_e],
                    out_e,
                )

        gr.Markdown(
            "_Tip: For NSFW realism try **Pony Realism SDXL**. "
            "On free Colab, unload/reload if you hit OOM; use SD 1.5 for lower VRAM._"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Unrestricted image generator")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--preload",
        choices=list(MODELS.keys()),
        default=None,
        help="Load a model at startup",
    )
    args = parser.parse_args()

    # Avoid HF symlink warnings in some containers
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if args.preload:
        print(load_model(args.preload))

    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
