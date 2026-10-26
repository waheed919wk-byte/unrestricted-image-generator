# Unrestricted Image Generator

Text-to-image + img2img with **safety checker disabled**. Adult / NSFW OK. **Adults only — no minors.**

Works on **any NVIDIA CUDA GPU**: Colab **T4**, **A100**, **L4**, local cards, etc.

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waheed919wk-byte/unrestricted-image-generator/blob/main/unrestricted_gen.ipynb)

1. Click **Open in Colab**
2. **Runtime → Change runtime type → GPU** (T4, **A100**, or L4 — all fine) → Save
3. Run the cell
4. Wait for a `https://….gradio.live` link and open it
5. When Status says **Ready**, Generate

> A100 is better than T4 (more VRAM, faster). You do **not** need to switch to T4.  
> If you see “No GPU”, step 2 was skipped.  
> First download is ~6–7 GB.

## Models

| Model | Notes |
|-------|--------|
| SDXL Base | Default, general quality |
| Pony Realism SDXL | NSFW-friendly |
| SD 1.5 | Use on low VRAM if you hit OOM |

On **A100** (24GB+), the UI allows up to **1536×1536**.

## Local (NVIDIA GPU)

```bash
pip install -r requirements.txt
# Install CUDA PyTorch from https://pytorch.org if needed
python unrestricted_gen.py --share
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No GPU` / CUDA error | Runtime → GPU (A100/T4/L4), then re-run |
| Session crashed | You may be on the wrong notebook — use the Colab link above, not Untitled Drive notebooks that call `os.kill` |
| Out of memory (OOM) | Switch to **SD 1.5**, or lower width/height |
| Share link fails | Re-run the cell; keep the Colab tab open |
