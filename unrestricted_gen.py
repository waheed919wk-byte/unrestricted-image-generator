# ============ INSTALL (stable versions) ============
!pip uninstall -y diffusers transformers accelerate torchao -q
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q diffusers==0.29.2 transformers==4.41.2 accelerate==0.30.1 gradio pillow

# ============ MAIN CODE ============
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import gradio as gr

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# Alternative: "John6666/pony-realism-v21-sdxl"

print("Loading models... (pehli dafa thoda time lagega)")

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    use_safetensors=True
).to("cuda")

pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    use_safetensors=True
).to("cuda")

pipe.enable_attention_slicing()
pipe_img2img.enable_attention_slicing()

def generate_image(prompt, negative_prompt, steps, guidance, width, height, seed):
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed != -1 else None
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=generator
    ).images[0]
    return image

def edit_image(init_image, prompt, negative_prompt, strength, steps, guidance, seed):
    if init_image is None:
        return None
    init_image = init_image.convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed != -1 else None
    image = pipe_img2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator
    ).images[0]
    return image

with gr.Blocks(title="Unrestricted Generator") as demo:
    gr.Markdown("# Fully Unrestricted Text-to-Image + Edit")
    
    with gr.Tabs():
        with gr.Tab("Generate"):
            prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Detailed prompt yahan likho...")
            neg = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, worst quality, deformed, ugly, bad anatomy, extra limbs, missing fingers, mutated hands, poorly drawn face, watermark, text, signature"
            )
            with gr.Row():
                steps = gr.Slider(15, 40, value=28, label="Steps")
                guidance = gr.Slider(4, 12, value=7, label="Guidance")
            with gr.Row():
                width = gr.Slider(768, 1024, value=1024, step=64, label="Width")
                height = gr.Slider(768, 1024, value=1024, step=64, label="Height")
            seed = gr.Number(value=-1, label="Seed (-1 = random)")
            btn = gr.Button("Generate", variant="primary")
            out = gr.Image(label="Result")
            btn.click(generate_image, [prompt, neg, steps, guidance, width, height, seed], out)

        with gr.Tab("Edit Image"):
            init = gr.Image(type="pil", label="Image upload karo")
            prompt_e = gr.Textbox(label="Edit Prompt", lines=4)
            neg_e = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, deformed, bad anatomy, extra limbs, missing fingers"
            )
            strength = gr.Slider(0.3, 0.85, value=0.6, label="Strength")
            with gr.Row():
                steps_e = gr.Slider(15, 40, value=28, label="Steps")
                guidance_e = gr.Slider(4, 12, value=7, label="Guidance")
            seed_e = gr.Number(value=-1, label="Seed")
            btn_e = gr.Button("Edit", variant="primary")
            out_e = gr.Image(label="Edited Result")
            btn_e.click(edit_image, [init, prompt_e, neg_e, strength, steps_e, guidance_e, seed_e], out_e)

demo.launch(share=True)
