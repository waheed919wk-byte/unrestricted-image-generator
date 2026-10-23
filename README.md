# Unrestricted Image Generator (Text-to-Image + Edit)

Fully unrestricted Stable Diffusion XL code for Google Colab.

- Safety checker **OFF**
- NSFW / adult content allowed
- Text-to-Image + Img2Img (edit)
- Gradio interface

## How to Run in Google Colab

1. Open this file in Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waheed919wk-byte/unrestricted-image-generator/blob/main/unrestricted_gen.ipynb)

2. Or manually:
   - Go to [Google Colab](https://colab.research.google.com)
   - Runtime → Change runtime type → **T4 GPU**
   - Copy the code from `unrestricted_gen.py` into a cell and run

## Notes
- First run downloads the model (~6-7 GB)
- Free Colab has time limits
- Use good negative prompts for better results

**Model used:** `stabilityai/stable-diffusion-xl-base-1.0`
