# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import tempfile
import pandas as pd

# ---------- Config ----------
IMG_TARGET_SIZE = (224, 224)  # change if your model expects another size
TOP_K = 5  

class_labels= [
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

# ---------- UI ----------
st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload your image. The app will load the model and predict the disease.")

uploaded_model = st.file_uploader("Upload Keras model file (.h5)", type=["h5"], key="model_uploader")
uploaded_image = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"], key="image_uploader")

# ---------- Helpers ----------
@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    """
    Save uploaded bytes to a temp file and load the Keras model from disk.
    compile=False is used to speed up load when training config is not needed.
    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    tmp_path = tmp_file.name
    try:
        tmp_file.write(model_bytes)
        tmp_file.flush()
        tmp_file.close()
        model = load_model(tmp_path, compile=False)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return model

def preprocess_pil(img: Image.Image, target_size=IMG_TARGET_SIZE):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_topk(model, img_array, top_k=TOP_K):
    preds = model.predict(img_array).flatten()
    top_idx = np.argsort(preds)[::-1][:top_k]
    topk_list = [(class_labels[i] if i < len(class_labels) else f"Class_{i}", float(preds[i])) for i in top_idx]
    return topk_list, preds

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

# ---------- App logic ----------
if uploaded_model is None:
    st.info("Please upload a `.h5` model file.")
if uploaded_image is None:
    st.info("Please upload an image to classify.")

if uploaded_model is not None and uploaded_image is not None:
    try:
        # Read model bytes (safe for various file-like objects)
        model_bytes = uploaded_model.read()
        model_size = len(model_bytes)
        st.write(f"**Model file:** {getattr(uploaded_model, 'name', 'uploaded_model.h5')}  —  {sizeof_fmt(model_size)}")

        if model_size > 300 * 1024 * 1024:
            st.warning("Uploaded model size exceeds 300 MB. Streamlit won't accept files larger than configured limit.")
        else:
            st.success("Model uploaded successfully.")

        with st.spinner("Loading model (this may take some time for large models)..."):
            model = load_model_from_bytes(model_bytes)
        st.success("Model loaded successfully.")

        # Read and validate image
        try:
            image_bytes = uploaded_image.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image. Please upload JPG or PNG.")
            raise
        except Exception:
            st.error("Error reading uploaded image.")
            raise

        st.image(pil_img, caption="Input image", use_column_width=True)

        # Preprocess and predict
        x = preprocess_pil(pil_img)
        with st.spinner("Running prediction..."):
            topk, full_preds = predict_topk(model, x, TOP_K)

        # Show only the top-1 prediction
        top_label, top_prob = topk[0]  # first element is the highest probability
        st.subheader("Top prediction")
        st.write(f"**{top_label}** — {top_prob:.4f}")

        # Checkbox to show full probability table (ONLY when clicked)
        if st.checkbox("Show full probabilities"):
            df = pd.DataFrame({
                "class_index": list(range(len(full_preds))),
                "class_label": [class_labels[i] if i < len(class_labels) else f"Class_{i}" for i in range(len(full_preds))],
                "probability": full_preds
            }).sort_values("probability", ascending=False).reset_index(drop=True)
            st.dataframe(df)

    except Exception as e:
        st.error("Error while loading model or predicting.")
        st.exception(e)
        st.write("Tips / Debugging:")
        st.write("- Ensure the `.h5` model was trained with the same class order as `class_labels`.")
        st.write("- If your model used different image preprocessing (mean subtraction / different size), update `preprocess_pil`.")
        st.write("- If model loading fails due to custom layers, provide `custom_objects` to `load_model`.")
        st.write("- For very large models consider converting to a TF SavedModel or quantizing to reduce memory footprint.")
else:
    st.write("")

st.markdown("---")
