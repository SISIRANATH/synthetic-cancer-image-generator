import os
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# -------------------------------------------------------------------
# Streamlit Page Settings
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🧫 Cloud Cancer Image Generator",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------------------------
# Hugging Face API Key
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Initialize Inference Client
# -------------------------------------------------------------------
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"     # you can change this

# -------------------------------------------------------------------
# Function to Generate Image
# -------------------------------------------------------------------
def generate_image(prompt: str) -> Image.Image:
    image = client.text_to_image(
        prompt,
        model=MODEL_ID,
    )
    return image

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.title("🧫 Cloud-Based Synthetic Cancer Cell Image Generator (HF InferenceClient)")
st.markdown("""
This version uses the **Hugging Face InferenceClient** for stable & fast cloud image generation.  
No local model needed — fully cloud powered!
""")

prompt = st.text_area(
    "🧠 Enter your prompt:",
    value="microscopic histopathology image of cancer cells, H&E stained, pink and purple"
)

if st.button("🚀 Generate Image"):
    with st.spinner("Generating image ... please wait ⏳"):
        try:
            image = generate_image(prompt)

            # Show image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Convert to bytes for download
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            st.download_button(
                "💾 Download Image",
                data=img_bytes,
                file_name="synthetic_cancer.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("⚙️ Powered by HuggingFace InferenceClient (Cloud)")
