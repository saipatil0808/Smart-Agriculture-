import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_PATH = "crop_disease_model.keras"
CLASS_PATH = "class_names.txt"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Smart Agriculture", page_icon="🌿", layout="centered")
st.title("🌿 Smart Agriculture: Crop Disease Detection")
st.write("Upload a leaf image to classify it with a trained CNN model.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.warning("The trained model files are not included yet. Train the model using train.py, then place crop_disease_model.keras and class_names.txt in this folder.")
    st.info("Dataset folder format: dataset/Class_Name/image.jpg")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    resized = image.resize(IMG_SIZE)
    arr = np.array(resized, dtype=np.float32) / 255.0
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(probs))
    st.subheader("Prediction")
    st.success(f"Predicted class: {class_names[idx]}")
    st.write(f"Model confidence: {float(probs[idx]) * 100:.2f}%")
    st.caption("This is an educational prediction, not a substitute for expert agricultural diagnosis.")
