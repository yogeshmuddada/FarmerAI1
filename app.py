# app.py
"""
Streamlit app to load a Keras model and predict plant disease from an uploaded image.

Usage:
    pip install streamlit tensorflow pillow numpy matplotlib
    streamlit run app.py

If your model path is different, change MODEL_PATH below or upload a model file using the optional upload control.
"""

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

st.title("Plant Disease Classifier (Keras .h5 model)")
st.write("Upload an image and the model will predict the class. The model should expect 224x224 RGB images normalized to [0,1].")

# ----- USER CONFIG -----
# Default model path (change if needed)
DEFAULT_MODEL_PATH = "E:/CNNModel.h5"

# Replace with your actual class names (the list from your prompt)
CLASS_LABELS = [
 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ----- MODEL LOADING -----
@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    model = load_model(path)
    return model

st.sidebar.header("Model options")
use_uploaded_model = st.sidebar.checkbox("Upload a .h5 model file instead of using default path", value=False)

model = None
if use_uploaded_model:
    uploaded_model_file = st.sidebar.file_uploader("Upload .h5 model file", type=["h5", "keras"], accept_multiple_files=False)
    if uploaded_model_file is not None:
        # save uploaded model to memory and load from bytes
        with st.spinner("Loading uploaded model..."):
            # Write bytes to temp file because load_model expects a path or h5 file-like object.
            # Some TF versions accept file-like; to be robust, write to a temp file.
            tmp_path = "/tmp/uploaded_model.h5"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_model_file.getbuffer())
            model = load_keras_model(tmp_path)
else:
    model_path = st.sidebar.text_input("Local model path", value=DEFAULT_MODEL_PATH)
    if model_path:
        try:
            with st.spinner("Loading model..."):
                model = load_keras_model(model_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load model from {model_path}: {e}")

if model is None:
    st.warning("Model not loaded yet. Provide a valid path or upload a model in the sidebar.")
    st.stop()

st.success("Model loaded successfully!")

# ----- IMAGE INPUT -----
st.header("Image input")
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    use_sample = st.button("Use example sample image")
    target_size = (224, 224)
    st.write(f"Model input size assumed: {target_size}")

# Sample image option: you can change this path to a local sample
SAMPLE_IMAGE_PATH = None  # set to a path string if you want a local file

img = None
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
elif use_sample:
    if SAMPLE_IMAGE_PATH:
        img = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    else:
        st.info("No local sample configured. Please upload an image or set SAMPLE_IMAGE_PATH in the script.")
        st.stop()

if img is not None:
    with col2:
        st.image(img, caption="Input image (original)", use_column_width=True)

    # Preprocess
    def preprocess_pil(pil_img, target_size):
        pil_resized = pil_img.resize(target_size)
        arr = np.asarray(pil_resized).astype("float32") / 255.0
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr]*3, axis=-1)
        arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, C)
        return arr

    x = preprocess_pil(img, target_size)

    # Predict
    with st.spinner("Predicting..."):
        preds = model.predict(x)
        preds = preds.reshape(-1)  # flat
        top_indices = preds.argsort()[::-1][:5]
        top_probs = preds[top_indices]
        top_labels = [CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Class_{i}" for i in top_indices]

    st.subheader("Prediction")
    st.write(f"**Top-1:** {top_labels[0]}  —  **prob:** {top_probs[0]:.4f}")

    # Show top-5 in a table
    import pandas as pd
    top_df = pd.DataFrame({
        "label": top_labels,
        "probability": top_probs
    })
    st.table(top_df)

    # Plot bar chart of top-5 probs
    fig, ax = plt.subplots()
    ax.barh(range(len(top_labels))[::-1], top_probs[::-1])
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels[::-1])
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 predictions")
    st.pyplot(fig)

    # Full probs (optionally)
    if st.checkbox("Show probabilities for all classes"):
        all_df = pd.DataFrame({
            "class_index": np.arange(len(preds)),
            "label": [CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Class_{i}" for i in range(len(preds))],
            "probability": preds
        }).sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(all_df)

    # Downloadable result text
    result_text = f"Top-1: {top_labels[0]} (prob {top_probs[0]:.4f})"
    st.download_button("Download prediction as text", data=result_text, file_name="prediction.txt", mime="text/plain")
else:
    st.info("Upload an image to get a prediction.")
