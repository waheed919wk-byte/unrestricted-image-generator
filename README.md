# Unrestricted Image Generator

Naya, simple setup. Safety checker **OFF**. Adult content OK. **Adults only.**

GPU: Colab **A100 / T4 / L4** (koi bhi NVIDIA CUDA)

## Colab (asaan tareeqa)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waheed919wk-byte/unrestricted-image-generator/blob/cursor/fresh-generator-cf16/unrestricted_gen.ipynb)

1. Link open karo  
2. **Runtime → Change runtime type → GPU → Save** (A100 rakho agar milay)  
3. Cell run karo  
4. `gradio.live` link open karo → Generate  

> Purani `Untitled1.ipynb` / Drive wali crashing notebooks mat chalao.

## Local

```bash
pip install -r requirements.txt
python app.py --share
```

## Files

| File | Kaam |
|------|------|
| `unrestricted_gen.ipynb` | Colab notebook |
| `app.py` | Gradio app |
| `requirements.txt` | Dependencies |
