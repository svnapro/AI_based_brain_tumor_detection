import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan to check if a brain tumor is detected.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    return model

model = load_model()

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    if prediction > 0.5:
        st.error(f"🚨 Tumor Detected with confidence {confidence*100:.2f}%")
    else:
        st.success(f"✅ No Tumor Detected with confidence {confidence*100:.2f}%")

st.caption("⚠️ For educational use only — not for medical diagnosis.")
