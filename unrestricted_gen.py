#!/usr/bin/env python3
"""
Unrestricted Text-to-Image + Edit (SDXL / SD1.5)
Safety checker OFF. Adult content allowed. Adults only — no minors.

Colab: Runtime → GPU (T4, A100, L4, …), then run the notebook.
Or locally: python unrestricted_gen.py --share
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import sys

# ---------------------------------------------------------------------------
# Soft imports so the script can print a clear error if deps are missing
# ---------------------------------------------------------------------------
def _die(msg: str, code: int = 1) -> None:
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(code)


try:
    import torch
except ImportError:
    _die("PyTorch not installed. On Colab just run the install cell first.")

try:
    import gradio as gr
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )
    from PIL import Image
except ImportError as e:
    _die(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")


MODELS = {
    "SDXL Base (general)": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "kind": "sdxl",
    },
    "Pony Realism SDXL (NSFW-friendly)": {
        "id": "John6666/pony-realism-v21main-sdxl",
        "kind": "sdxl",
    },
    "SD 1.5 (lighter VRAM)": {
        "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "kind": "sd15",
    },
}

DEFAULT_MODEL = "SDXL Base (general)"

DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, deformed, ugly, bad anatomy, "
    "extra limbs, missing fingers, mutated hands, poorly drawn face, "
    "watermark, text, signature, child, underage, loli, young"
)

_state: dict = {"pipe": None, "pipe_i2i": None, "model_key": None, "kind": None, "device": None}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "No CUDA GPU"
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    total_gb = props.total_memory / 1024**3
    return f"{name} ({total_gb:.0f} GB)"


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def _clear_gpu() -> None:
    _state["pipe"] = None
    _state["pipe_i2i"] = None
    _state["model_key"] = None
    _state["kind"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pipeline_from_components(cls, pipe):
    """Build a sibling pipeline sharing weights (filters incompatible kwargs)."""
    params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in pipe.components.items() if k in params and v is not None}
    return cls(**kwargs)


def load_model(model_key: str, progress=gr.Progress(track_tqdm=True)):
    device = get_device()
    if device != "cuda":
        raise gr.Error(
            "No GPU detected. In Colab: Runtime → Change runtime type → "
            "GPU (T4, A100, L4, or any NVIDIA) → Save, then Run all."
        )

    if model_key not in MODELS:
        raise gr.Error(f"Unknown model: {model_key}")

    if _state["model_key"] == model_key and _state["pipe"] is not None:
        return f"Already loaded: {model_key}"

    progress(0, desc=f"Downloading / loading {model_key}…")
    _clear_gpu()

    info = MODELS[model_key]
    model_id = info["id"]
    kind = info["kind"]
    dtype = torch.float16

    try:
        if kind == "sdxl":
            load_kwargs = dict(torch_dtype=dtype, use_safetensors=True)
            # add_watermarker exists on some diffusers versions only
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id, variant="fp16", add_watermarker=False, **load_kwargs
                )
            except TypeError:
                try:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_id, variant="fp16", **load_kwargs
                    )
                except Exception:
                    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)
            except Exception:
                # community checkpoints may lack fp16 variant
                try:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_id, add_watermarker=False, **load_kwargs
                    )
                except TypeError:
                    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)

            pipe = pipe.to(device)
            pipe_i2i = _pipeline_from_components(StableDiffusionXLImg2ImgPipeline, pipe)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
            pipe = pipe.to(device)
            pipe_i2i = _pipeline_from_components(StableDiffusionImg2ImgPipeline, pipe)
    except Exception as e:
        _clear_gpu()
        raise gr.Error(
            f"Failed to load {model_key}: {e}\n"
            "Try SD 1.5 (lighter VRAM), or Runtime → Disconnect and delete runtime, then retry."
        ) from e

    pipe.enable_attention_slicing()
    pipe_i2i.enable_attention_slicing()
    for p in (pipe, pipe_i2i):
        try:
            p.enable_vae_slicing()
        except Exception:
            pass
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    _state["pipe"] = pipe
    _state["pipe_i2i"] = pipe_i2i
    _state["model_key"] = model_key
    _state["kind"] = kind
    _state["device"] = device

    used = torch.cuda.memory_allocated() / 1024**3
    return f"Ready: {model_key} on {gpu_info()} — using {used:.1f} GB"


def _generator(seed: int):
    device = _state.get("device") or get_device()
    if seed is None or int(seed) < 0:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))


def _ensure_loaded():
    if _state["pipe"] is None:
        raise gr.Error("Model not loaded yet. Click Load Model (or wait for auto-load).")


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
    w, h = int(width), int(height)
    if kind == "sd15":
        w = max(384, min(768, (w // 8) * 8))
        h = max(384, min(768, (h // 8) * 8))
    else:
            cap = 1536 if gpu_total_gb() >= 24 else 1280
            w = max(768, min(cap, (w // 64) * 64))
            h = max(768, min(cap, (h // 64) * 64))

    n = max(1, min(4, int(num_images)))
    images = []
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else None

    for i in range(n):
        s = (base_seed + i) if base_seed is not None else -1
        out = _state["pipe"](
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=w,
            height=h,
            generator=_generator(s),
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


def build_ui(status_text: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="Unrestricted Image Generator") as demo:
        gr.Markdown(
            """
# Unrestricted Image Generator
Text-to-image + img2img · **Safety checker OFF** · Adult content OK · **Adults only**

**Colab:** Runtime → **GPU** (T4 / **A100** / L4 all work). If Status is not Ready, click **Load / Switch Model**.
            """
        )

        # A100 (40GB+) can push higher resolution than T4
        max_side = 1536 if gpu_total_gb() >= 24 else 1280
        default_side = 1024

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Model",
            )
            load_btn = gr.Button("Load / Switch Model", variant="primary")
        status = gr.Textbox(
            label="Status",
            interactive=False,
            value=status_text
            or ("Ready: " + _state["model_key"] if _state.get("pipe") else "Click Load / Switch Model"),
        )
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
                    width = gr.Slider(512, max_side, value=default_side, step=64, label="Width")
                    height = gr.Slider(512, max_side, value=default_side, step=64, label="Height")
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

    return demo


def preload(model_key: str | None = None) -> str:
    """Load weights in the notebook/terminal before Gradio starts (clear progress)."""
    key = model_key or DEFAULT_MODEL
    print(f"Loading {key}… (first time downloads several GB)")

    class _Prog:
        def __call__(self, *args, **kwargs):
            desc = kwargs.get("desc") or (args[1] if len(args) > 1 else "")
            if desc:
                print(desc)

    return load_model(key, progress=_Prog())


def main():
    global DEFAULT_MODEL

    parser = argparse.ArgumentParser(description="Unrestricted image generator")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-autoload", action="store_true", help="Do not load model on start")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help="Default model",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    device = get_device()
    print(f"Device: {device} | {gpu_info()}")
    if device != "cuda":
        print(
            "WARNING: No CUDA GPU. The UI will open, but Load Model will fail "
            "until you use a GPU runtime (Colab: T4, A100, L4, …)."
        )

    DEFAULT_MODEL = args.model

    status_text = None
    if not args.no_autoload and device == "cuda":
        try:
            status_text = preload(args.model)
            print(status_text)
        except Exception as e:
            print(f"Auto-load failed: {e}")
            print("UI will still open — use Load / Switch Model.")

    demo = build_ui(status_text=status_text)
    print("Launching Gradio…")
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
