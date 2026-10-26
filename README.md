# Unrestricted Image Generator

Text-to-image + img2img with **safety checker disabled**. Adult / NSFW content allowed. **Adults only — no minors.**

Built on Diffusers + Gradio. Works on **Google Colab (T4)** or any local NVIDIA GPU.

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waheed919wk-byte/unrestricted-image-generator/blob/main/unrestricted_gen.ipynb)

1. Runtime → Change runtime type → **T4 GPU**
2. Run the install cell, then the launch cell
3. Open the public Gradio link
4. **Load Model** → Generate or Edit

## Models

| Model | Notes |
|-------|--------|
| SDXL Base | General quality, ~6–7 GB download |
| Pony Realism SDXL | NSFW-friendly community checkpoint |
| SD 1.5 | Lighter VRAM; safety checker forced off |

## Local (CUDA)

```bash
pip install -r requirements.txt
# Install CUDA build of PyTorch from https://pytorch.org if needed
python unrestricted_gen.py --share
```

Options:

```bash
python unrestricted_gen.py --share --port 7860
python unrestricted_gen.py --preload "SDXL Base (general)"
```

## Notes

- First run downloads the chosen model weights
- Free Colab has session time limits
- One pipeline load shared for txt2img + img2img (saves VRAM vs loading twice)
- Attention / VAE slicing enabled; xFormers used when available

## Files

- `unrestricted_gen.py` — Gradio app
- `unrestricted_gen.ipynb` — Colab launcher
- `requirements.txt` — Python deps
