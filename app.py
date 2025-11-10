import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

st.set_page_config(page_title="🧫 Synthetic Cancer Image Generator", page_icon="🧬", layout="wide")

@st.cache_resource
def load_model():
    model_id = "stabilityai/sd-turbo"      # ⚡ super-light & fast SD model
    device = "cpu"                         # CPU only, works everywhere

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    return pipe


# ---------------------- UI -------------------------
st.title("🧫 Synthetic Cancer Cell Image Generator")
st.markdown("""
Generate **synthetic histopathology-like cancer images** using a **lightweight Stable Diffusion Turbo model**.
""")

default_prompt = "microscopic H&E stained image of invasive carcinoma cells, histopathology, pink and purple tones, realistic"
prompt = st.text_area("🧠 Enter your prompt:", value=default_prompt, height=80)

steps = st.slider("🌀 Inference Steps", 1, 10, 4, 1)
seed = st.number_input("🌱 Random Seed (-1 = random)", value=42)

if st.button("🚀 Generate Image"):
    st.write("⏳ Generating image…")

    pipe = load_model()

    # seed control
    generator = torch.Generator("cpu").manual_seed(int(seed)) if seed >= 0 else None

    start = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        width=256,
        height=256,
        generator=generator
    ).images[0]

    st.image(image, caption="Generated image", use_column_width=True)

    st.download_button(
        "💾 Download Image",
        data=image.tobytes(),
        file_name="synthetic_cancer.png",
        mime="image/png"
    )

    st.success(f"Done in {time.time() - start:.2f} seconds")

st.markdown("---")
st.markdown("⚙️ Powered by Streamlit + Diffusers (No Docker Required)")
