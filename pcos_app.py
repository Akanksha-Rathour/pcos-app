import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import tempfile
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Classifier",
    page_icon="🔬",
    layout="centered",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    .pcos    { background: #FDECEA; color: #B71C1C; border: 1px solid #EF9A9A; }
    .nonpcos { background: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7; }
    .metric-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    # Update this path to wherever you saved xgboost_model.pkl
    xgb = joblib.load("xgboost_model.pkl")
    return cnn, xgb

# ─────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────
def extract_cnn_features(pil_image, cnn_model):
    img = pil_image.convert("RGB").resize((299, 299))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = cnn_model.predict(arr, verbose=0)
    return features.flatten()


def extract_glcm_features(pil_image):
    img_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast    = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy      = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])


def predict(pil_image, cnn_model, xgb_model):
    cnn_feat  = extract_cnn_features(pil_image, cnn_model)
    glcm_feat = extract_glcm_features(pil_image)
    combined  = np.hstack([cnn_feat, glcm_feat]).reshape(1, -1)

    label      = xgb_model.predict(combined)[0]          # 0 = PCOS, 1 = Non-PCOS  ← matches your class order
    proba      = xgb_model.predict_proba(combined)[0]
    confidence = proba[label]
    return int(label), float(confidence), proba


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🔬 PCOS Ultrasound Classifier")
st.markdown(
    "Upload an **ovarian ultrasound image** and the model will classify it as "
    "**PCOS** or **Non-PCOS** using InceptionV3 + GLCM texture features with an XGBoost classifier."
)

st.divider()

# Sidebar: model info
with st.sidebar:
    st.header("Model Info")
    st.markdown("""
    **Architecture**
    - Feature extractor: InceptionV3 (ImageNet)
    - Texture features: GLCM (4 features)
    - Classifier: XGBoost (300 estimators)

    **Reported Performance**
    - Test Accuracy: ~91%

    **Feature Vector**
    - CNN: 2048 dims
    - GLCM: 4 dims
    - Combined: 2052 dims

    **Classes**
    - `0` → PCOS
    - `1` → Non-PCOS
    """)

# File uploader
uploaded = st.file_uploader(
    "Choose an ultrasound image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    help="Accepts PNG, JPG, BMP, TIFF",
)

if uploaded is not None:
    pil_img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with col2:
        st.markdown("**Image details**")
        st.markdown(f"- Filename: `{uploaded.name}`")
        st.markdown(f"- Size: `{pil_img.size[0]} × {pil_img.size[1]} px`")
        st.markdown(f"- Mode: `{pil_img.mode}`")

    st.divider()

    if st.button("🔍 Run Classification", type="primary", use_container_width=True):
        with st.spinner("Loading models and extracting features…"):
            try:
                cnn_model, xgb_model = load_models()
            except FileNotFoundError:
                st.error(
                    "❌ `xgboost_model.pkl` not found. "
                    "Place it in the same directory as this script."
                )
                st.stop()

        with st.spinner("Classifying…"):
            label, confidence, proba = predict(pil_img, cnn_model, xgb_model)

        # ── Result ──────────────────────────────
        st.subheader("Result")

        if label == 0:
            st.markdown(
                f'<div class="result-box pcos">⚠️ PCOS Detected — {confidence*100:.1f}% confidence</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box nonpcos">✅ Non-PCOS — {confidence*100:.1f}% confidence</div>',
                unsafe_allow_html=True,
            )

        # ── Probability bar chart ────────────────
        st.markdown("**Class probabilities**")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("PCOS",     f"{proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
        with prob_col2:
            st.metric("Non-PCOS", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

        # ── GLCM features breakdown ──────────────
        with st.expander("📊 GLCM texture features (interpretable)"):
            glcm = extract_glcm_features(pil_img)
            feat_names = ["Contrast", "Correlation", "Energy", "Homogeneity"]
            for name, val in zip(feat_names, glcm):
                st.markdown(
                    f'<div class="metric-card"><b>{name}</b>: {val:.6f}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption(
            "⚠️ This tool is for **research and educational purposes only** "
            "and is not a substitute for clinical diagnosis."
        )

else:
    st.info("👆 Upload an ultrasound image above to get started.")
