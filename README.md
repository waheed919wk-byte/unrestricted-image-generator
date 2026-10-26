# Unrestricted Image Generator

Text-to-image + img2img with **safety checker disabled**. Adult / NSFW OK. **Adults only — no minors.**

## Run on Google Colab (easiest)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waheed919wk-byte/unrestricted-image-generator/blob/cursor/improve-unrestricted-generator-cf16/unrestricted_gen.ipynb)

1. Click **Open in Colab**
2. **Runtime → Change runtime type → T4 GPU → Save**
3. **Runtime → Run all**
4. Wait for a `https://….gradio.live` link and open it
5. Wait until Status says **Ready**, then Generate

> If you see “No GPU”, you skipped step 2.  
> If the link never appears, re-run the last cell. First download is ~6–7 GB.

## Models

| Model | Notes |
|-------|--------|
| SDXL Base | Default, general quality |
| Pony Realism SDXL | NSFW-friendly |
| SD 1.5 | Use this if you hit out-of-memory |

## Local (NVIDIA GPU)

```bash
pip install -r requirements.txt
# Install CUDA PyTorch from https://pytorch.org if needed
python unrestricted_gen.py --share
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No GPU` / CUDA error | Colab → Runtime → T4 GPU, then Run all |
| Out of memory (OOM) | Switch model to **SD 1.5**, or lower width/height to 768 |
| Share link fails | Re-run the launch cell; check Colab is still connected |
| Stuck on download | Wait; or Runtime → Disconnect and delete runtime → Run all again |
