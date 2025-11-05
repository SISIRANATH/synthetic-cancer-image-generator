import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import time

st.set_page_config(page_title="🧫 Synthetic Cancer Image Generator", page_icon="🧬", layout="wide")

@st.cache_resource
def load_model():
    model_id = "rupeshs/LCM-runwayml-stable-diffusion-v1-5"
    device = "cpu"   # ✅ force CPU (no GPU needed)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Enable lightweight settings
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    return pipe

# Title + description
st.title("🧫 Synthetic Cancer Cell Image Generator")
st.markdown("""
Generate **synthetic histopathology-like (cancer) images** using a **pretrained Stable Diffusion model (no dataset needed)**.
This demo uses the open-source model [`rupeshs/LCM-runwayml-stable-diffusion-v1-5`](https://huggingface.co/rupeshs/LCM-runwayml-stable-diffusion-v1-5).
""")

# Prompt input
default_prompt = "microscopic H&E stained image of invasive ductal carcinoma cells, histopathology, pink and purple tones, realistic"
prompt = st.text_area("🧠 Enter your prompt:", value=default_prompt, height=80)

guidance = st.slider("🎛️ Guidance Scale", 2.0, 12.0, 7.5, 0.5)
steps = st.slider("🌀 Inference Steps", 10, 50, 25, 1)
seed = st.number_input("🌱 Random Seed (-1 = random)", value=42, step=1)

generate = st.button("🚀 Generate Synthetic Image")

if generate:
    st.write("⏳ Generating image... Please wait.")
    start = time.time()

    pipe = load_model()
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed) if seed >= 0 else torch.seed())

    image = pipe(
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=min(steps, 20),
        width=256, height=256,       # smaller image size
        generator=generator
    ).images[0]


    elapsed = time.time() - start
    st.image(image, caption=f"Generated image in {elapsed:.2f}s", use_column_width=True)
    st.download_button("💾 Download Image", data=image.tobytes(), file_name="synthetic_cancer.png", mime="image/png")

st.markdown("---")
st.markdown("⚙️ Built with [Streamlit](https://streamlit.io) and [Diffusers](https://huggingface.co/docs/diffusers).")
