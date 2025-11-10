import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

st.set_page_config(page_title="🧫 Cloud Cancer Image Generator", page_icon="🧬", layout="wide")

# 🔐 Load Hugging Face API Key from Streamlit Secrets
HF_API_KEY = st.secrets["HF_TOKEN"]

# Initialize HF client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# UI
st.title("🧫 Cloud-Based Synthetic Cancer Cell Image Generator")
st.markdown("""
This version uses **Hugging Face Inference API** (Cloud).  
No local model is loaded — everything runs on Hugging Face servers.
""")

prompt = st.text_area(
    "🧠 Enter your prompt:",
    value="microscopic histopathology image of cancer cells, H&E stained, pink and purple"
)

if st.button("🚀 Generate Image"):
    with st.spinner("Generating image... please wait ⏳"):
        try:
            image = client.text_to_image(
                prompt,
                model=MODEL_NAME
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            st.image(image, caption="Generated Image", use_column_width=True)

            st.download_button(
                "💾 Download Image",
                data=image_bytes,
                file_name="synthetic_cancer.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("⚙️ Powered by Hugging Face Inference API (Cloud)")
