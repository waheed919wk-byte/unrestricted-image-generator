#!/usr/bin/env python3
"""Unrestricted text-to-image + img2img. Safety checker OFF. Adults only."""

from __future__ import annotations

import argparse
import gc
import inspect
import os

import gradio as gr
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# NSFW-friendly alternative (uncomment to use):
# MODEL_ID = "John6666/pony-realism-v21main-sdxl"

NEG = (
    "blurry, low quality, worst quality, deformed, ugly, bad anatomy, "
    "extra limbs, missing fingers, watermark, text, child, underage, loli"
)

pipe = None
pipe_i2i = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gpu_label() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} ({gb:.0f} GB)"


def _from_components(cls, base):
    params = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in base.components.items() if k in params and v is not None}
    return cls(**kw)


def load_model():
    global pipe, pipe_i2i
    if _device() != "cuda":
        raise RuntimeError(
            "GPU chahiye. Colab: Runtime → Change runtime type → GPU (A100/T4/L4) → Save"
        )

    print(f"GPU: {_gpu_label()}")
    print(f"Loading {MODEL_ID} … (pehli dafa download hoga)")

    pipe = None
    pipe_i2i = None
    gc.collect()
    torch.cuda.empty_cache()

    kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
    try:
        p = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, variant="fp16", add_watermarker=False, **kwargs
        )
    except TypeError:
        try:
            p = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID, variant="fp16", **kwargs
            )
        except Exception:
            p = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **kwargs)
    except Exception:
        p = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **kwargs)

    p = p.to("cuda")
    p.enable_attention_slicing()
    try:
        p.enable_vae_slicing()
    except Exception:
        pass

    pipe = p
    pipe_i2i = _from_components(StableDiffusionXLImg2ImgPipeline, p)
    pipe_i2i.enable_attention_slicing()

    used = torch.cuda.memory_allocated() / 1024**3
    msg = f"Ready — {_gpu_label()} — model {used:.1f} GB"
    print(msg)
    return msg


def generate(prompt, negative, steps, guidance, width, height, seed):
    if pipe is None:
        raise gr.Error("Model load nahi hua. Cell dubara run karo.")
    if not (prompt or "").strip():
        raise gr.Error("Prompt likho.")

    w = max(768, min(1536, (int(width) // 64) * 64))
    h = max(768, min(1536, (int(height) // 64) * 64))
    gen = None
    if seed is not None and int(seed) >= 0:
        gen = torch.Generator("cuda").manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative or "",
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=w,
        height=h,
        generator=gen,
    )
    return out.images[0]


def edit(image, prompt, negative, strength, steps, guidance, seed):
    if pipe_i2i is None:
        raise gr.Error("Model load nahi hua.")
    if image is None:
        raise gr.Error("Image upload karo.")
    if not (prompt or "").strip():
        raise gr.Error("Edit prompt likho.")

    gen = None
    if seed is not None and int(seed) >= 0:
        gen = torch.Generator("cuda").manual_seed(int(seed))

    out = pipe_i2i(
        prompt=prompt,
        negative_prompt=negative or "",
        image=image.convert("RGB"),
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=gen,
    )
    return out.images[0]


def build_ui(status: str = "") -> gr.Blocks:
    with gr.Blocks(title="Unrestricted Image Generator") as demo:
        gr.Markdown(
            f"""
# Unrestricted Image Generator
Safety checker **OFF** · Adult OK · **Adults only** (no minors)

**GPU:** {_gpu_label()}
            """
        )
        gr.Textbox(value=status or "Model ready", label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Detailed prompt…")
                neg = gr.Textbox(label="Negative", value=NEG, lines=2)
                with gr.Row():
                    steps = gr.Slider(15, 50, value=28, step=1, label="Steps")
                    guidance = gr.Slider(1, 12, value=7, step=0.5, label="Guidance")
                with gr.Row():
                    width = gr.Slider(768, 1536, value=1024, step=64, label="Width")
                    height = gr.Slider(768, 1536, value=1024, step=64, label="Height")
                    seed = gr.Number(value=-1, label="Seed (-1 random)", precision=0)
                btn = gr.Button("Generate", variant="primary")
                out = gr.Image(label="Result", height=512)
                btn.click(generate, [prompt, neg, steps, guidance, width, height, seed], out)

            with gr.Tab("Edit"):
                img = gr.Image(type="pil", label="Upload", height=320)
                prompt_e = gr.Textbox(label="Edit prompt", lines=3)
                neg_e = gr.Textbox(label="Negative", value=NEG, lines=2)
                with gr.Row():
                    strength = gr.Slider(0.2, 0.95, value=0.55, step=0.05, label="Strength")
                    steps_e = gr.Slider(15, 50, value=28, step=1, label="Steps")
                    guidance_e = gr.Slider(1, 12, value=7, step=0.5, label="Guidance")
                seed_e = gr.Number(value=-1, label="Seed (-1 random)", precision=0)
                btn_e = gr.Button("Edit", variant="primary")
                out_e = gr.Image(label="Result", height=512)
                btn_e.click(
                    edit, [img, prompt_e, neg_e, strength, steps_e, guidance_e, seed_e], out_e
                )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    status = load_model()
    demo = build_ui(status)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
