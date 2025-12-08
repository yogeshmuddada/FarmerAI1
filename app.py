# app.py
import os
import io
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError

# ---------- Config ----------
IMG_TARGET_SIZE = (224, 224)
TOP_K = 5

class_labels = [
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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Plant Disease Predictor — Upload Model & Image", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload your image. The app will predict the disease and provide a short AI explanation.")

uploaded_model = st.file_uploader("Upload Keras model file (.h5)", type=["h5"], key="model_uploader")
uploaded_image = st.file_uploader("Upload an image (jpg / png)", type=["jpg", "jpeg", "png"], key="image_uploader")


if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    """
    Save uploaded bytes to a temp file and load the Keras model from disk.
    compile=False speeds up load when training config is not needed.
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


if uploaded_model is None:
    st.info("Please upload a `.h5` model file.")
if uploaded_image is None:
    st.info("Please upload an image to classify.")

if uploaded_model is not None and uploaded_image is not None:
    try:
        
        model_bytes = uploaded_model.read()
        model_size = len(model_bytes)
        st.write(f"**Model file:** {getattr(uploaded_model, 'name', 'uploaded_model.h5')}  —  {sizeof_fmt(model_size)}")

        if model_size > 300 * 1024 * 1024:
            st.warning("Uploaded model size exceeds 300 MB. Streamlit may reject very large files.")
        else:
            st.success("Model uploaded successfully.")

        with st.spinner("Loading model (this may take some time for large models)..."):
            model = load_model_from_bytes(model_bytes)
        st.success("Model loaded successfully.")

        
        try:
            image_bytes = uploaded_image.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # validate file
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image. Please upload JPG or PNG.")
            raise
        except Exception:
            st.error("Error reading uploaded image.")
            raise

        st.image(pil_img, caption="Input image", use_column_width=True)

        
        x = preprocess_pil(pil_img)
        with st.spinner("Running prediction..."):
            topk, full_preds = predict_topk(model, x, TOP_K)

       
        top_label, top_prob = topk[0]  
        st.subheader("Top prediction")
        st.write(f"**{top_label}**")

        
        if st.checkbox("Show full probabilities"):
            df = pd.DataFrame({
                "class_index": list(range(len(full_preds))),
                "class_label": [class_labels[i] if i < len(class_labels) else f"Class_{i}" for i in range(len(full_preds))],
                "probability": full_preds
            }).sort_values("probability", ascending=False).reset_index(drop=True)
            st.dataframe(df)

        # ----------------- AI Explanation (Gemini) -----------------
        if not os.environ.get("GOOGLE_API_KEY"):
            st.warning("GOOGLE_API_KEY not found in .streamlit/secrets.toml. AI explanation will be skipped.")
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_available = True
            except Exception as e:
                llm_available = False
                st.error(f"Could not import ChatGoogleGenerativeAI: {e}. Install the appropriate package to enable AI explanations.")

            if llm_available:
                
                prompt = (
                f"""
You are an agricultural expert. The detected plant condition is: '{predicted_class}'.

Your task:
1. If the class indicates a DISEASE, provide:
   - A simple explanation in very easy layman language that farmers can understand (max 100 words)
   - Precautions farmers should take for that crop and disease
   - Recommended pesticides (common, widely available options)
   - Organic or natural methods to control or cure the disease
   - Tips to improve yield

2. If the class is 'Healthy', provide:
   - Simple confirmation that the plant is healthy
   - General care tips to maintain good yield

Supported classes:
Corn: Common Rust, Gray Leaf Spot, Leaf Blight, Healthy
Potato: Early Blight, Late Blight, Healthy
Rice: Brown Spot, Hispa, Leaf Blast, Healthy
Wheat: Brown Rust, Yellow Rust, Healthy

Your answer must be short, clear, structured, and farmer-friendly.
""")

                with st.spinner("Generating short explanation with AI..."):
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                        result = llm.invoke(prompt)
                        ai_response = getattr(result, "content", None) or str(result)
                    except Exception as e:
                        ai_response = f"AI generation failed: {e}"

                st.subheader("AI Explanation")
                st.write(ai_response)

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





