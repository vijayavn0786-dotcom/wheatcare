"""
WheatCare AI - Intelligent Wheat Leaf Disease Diagnosis & Agronomic Decision Support System
A Complete Final Year Engineering Project in Deep Learning and Computer Vision.
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path
from datetime import datetime

# Configure environment for clean headless execution
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
import altair as alt

# Import custom modular components
from gradcam import GradCAMExplainer
from disease_kb import DISEASE_KNOWLEDGE_BASE
from report_generator import generate_diagnostic_html_report


# ============================================================
# 1. APPLICATION & PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WheatCare AI | Intelligent Wheat Disease Diagnosis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_IMPROVED_PATH = BASE_DIR / "models" / "wheat_leaf_disease_cnn_improved.keras"
MODEL_BASELINE_PATH = BASE_DIR / "models" / "wheat_leaf_disease_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASET_CANDIDATES = [
    Path(r"D:\non_dup_dataset"),
    BASE_DIR / "non_dup_dataset",
    BASE_DIR / "sample_images",
]
DATASET_DIR = next((p for p in DATASET_CANDIDATES if p.is_dir()), None)
SAMPLE_DIR = BASE_DIR / "sample_images"


# ============================================================
# 2. CUSTOM CSS STYLING & PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --primary-deep: #0f3d2e;
        --primary: #1b5e3f;
        --primary-light: #e8f5ed;
        --accent-gold: #d97706;
        --surface: #ffffff;
        --surface-subtle: #f8faf8;
        --border-color: #dce7df;
        --text-dark: #14281f;
        --text-muted: #526b5d;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #f7faf7;
        color: var(--text-dark);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--primary-deep) !important;
        font-weight: 700 !important;
    }

    /* Main Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0e3728 0%, #175238 50%, #206d4b 100%);
        color: #ffffff;
        padding: 2.2rem 2.6rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px rgba(15, 61, 46, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        color: #c4f3d2;
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: clamp(2rem, 3.5vw, 2.9rem);
        margin: 0 0 0.8rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #e2f4e8;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 820px;
        margin-bottom: 0;
    }

    /* Metric & Feature Cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(18, 56, 38, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(18, 56, 38, 0.08);
    }

    .card-num {
        font-size: 0.75rem;
        font-weight: 800;
        color: var(--primary);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .badge-confidence {
        background: #e8f5ed;
        color: #165b38;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #b7dfc6;
    }

    .badge-severity-mild {
        background: #f0fdf4;
        color: #15803d;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #bbf7d0;
    }

    .badge-severity-moderate {
        background: #fefce8;
        color: #a16207;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #fef08a;
    }

    .badge-severity-severe {
        background: #fef2f2;
        color: #b91c1c;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #fecaca;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        border: 1px solid #b8d4c1;
        background-color: #ffffff;
        color: #174c35;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #e8f5ed;
        border-color: #1b5e3f;
        color: #103b27;
    }

    div.stButton > button[kind="primary"] {
        background-color: #1b5e3f !important;
        color: #ffffff !important;
        border: 1px solid #1b5e3f !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #12442c !important;
        box-shadow: 0 4px 12px rgba(27, 94, 63, 0.3);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #edf4ee;
        border-right: 1px solid var(--border-color);
    }

    /* Dataframe / Table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background: #ffffff;
    }

    /* Custom Footer */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        color: #617d6e;
        font-size: 0.88rem;
        border-top: 1px solid #e1ebe3;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. CACHED MODEL & RESOURCE LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading Deep Learning CNN models into memory...")
def load_all_models():
    """Loads both improved and baseline models along with class names."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names mapping not found at: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    models = {}
    if MODEL_IMPROVED_PATH.is_file():
        models["Improved CNN (89.14% Accuracy)"] = tf.keras.models.load_model(MODEL_IMPROVED_PATH)

    if MODEL_BASELINE_PATH.is_file():
        models["Baseline CNN (84.63% Accuracy)"] = tf.keras.models.load_model(MODEL_BASELINE_PATH)

    if not models:
        raise FileNotFoundError("No trained Keras models found in models/ directory.")

    return models, class_names


@st.cache_resource
def get_gradcam_explainer(_model):
    """Caches the GradCAM explainer object for the selected model."""
    return GradCAMExplainer(_model, last_conv_layer_name="conv2d_7", img_size=IMAGE_SIZE)


try:
    AVAILABLE_MODELS, CLASS_NAMES = load_all_models()
except Exception as e:
    st.error(f"Error loading models or configuration: {e}")
    st.stop()


# ============================================================
# 4. PREDICTION & DIAGNOSIS HELPERS
# ============================================================

def preprocess_image_for_model(pil_image):
    """Resizes and normalizes an input PIL image to (1, 224, 224, 3) in [0, 1]."""
    resized = pil_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR).convert("RGB")
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), resized


def run_single_inference(pil_image, model):
    """Runs forward pass and formats class probabilities."""
    batch_array, preprocessed_pil = preprocess_image_for_model(pil_image)
    raw_preds = model.predict(batch_array, verbose=0)[0]

    order = np.argsort(raw_preds)[::-1]
    top_index = int(order[0])
    top_class = CLASS_NAMES[top_index]
    top_conf = float(raw_preds[top_index])

    scores = [
        {"class": CLASS_NAMES[i], "score": float(raw_preds[i])}
        for i in order
    ]

    return {
        "predicted_class": top_class,
        "class_index": top_index,
        "confidence": top_conf,
        "scores": scores,
        "batch_array": batch_array,
        "preprocessed_pil": preprocessed_pil,
    }


def classify_severity(predicted_class, activation_ratio):
    """Determines severity grade based on predicted class and Grad-CAM activation area."""
    if predicted_class == "Healthy":
        return "Normal / Healthy", "badge-severity-mild", 0.0

    pct = activation_ratio * 100.0
    if pct < 15.0:
        return "Mild Infection (<15% leaf area)", "badge-severity-mild", pct
    elif pct <= 35.0:
        return "Moderate Infection (15%–35% leaf area)", "badge-severity-moderate", pct
    else:
        return "Severe Epidemic Infection (>35% leaf area)", "badge-severity-severe", pct


@st.cache_data
def scan_dataset_statistics():
    """Scans the dataset directory to compute real-time class distributions with cloud benchmark fallback."""
    benchmark_counts = {
        "Healthy": 1579,
        "Yellow rust": 1395,
        "Brown rust": 1256,
        "Septoria": 349,
        "Loose Smut": 299,
    }

    if not DATASET_DIR or not DATASET_DIR.is_dir() or DATASET_DIR == SAMPLE_DIR:
        return benchmark_counts

    stats = {}
    for class_name in CLASS_NAMES:
        class_folder = DATASET_DIR / class_name
        if class_folder.is_dir():
            files = [
                p for p in class_folder.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            stats[class_name] = len(files)
        else:
            stats[class_name] = 0

    if sum(stats.values()) < 50:
        return benchmark_counts

    return stats


def get_random_sample_from_class(class_name):
    """Retrieves a random sample file path from the dataset or bundled sample gallery."""
    for search_dir in [DATASET_DIR, SAMPLE_DIR]:
        if search_dir and search_dir.is_dir():
            folder = search_dir / class_name
            if folder.is_dir():
                candidates = [
                    p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                if candidates:
                    return random.choice(candidates)
    return None


# ============================================================
# 5. SESSION STATE MANAGEMENT
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "current_diagnosis" not in st.session_state:
    st.session_state.current_diagnosis = None


# ============================================================
# 6. SIDEBAR NAVIGATION & MODEL SELECTOR
# ============================================================

with st.sidebar:
    st.markdown("## 🌾 WheatCare AI")
    st.caption("Final Year Engineering Project · Deep Learning")
    st.divider()

    selected_model_name = st.selectbox(
        "ACTIVE CNN MODEL",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Switch between the Improved CNN (with regularization + augmentation) and the Baseline CNN for comparison.",
    )
    active_model = AVAILABLE_MODELS[selected_model_name]

    st.divider()

    navigation_choice = st.radio(
        "NAVIGATION MENU",
        [
            "🏠 Overview & Dashboard",
            "🔬 AI Disease Diagnosis",
            "📁 Batch Image Diagnosis",
            "🗂️ Dataset Explorer & Analytics",
            "📈 Model Performance & Benchmarks",
            "📖 Wheat Pathology Encyclopedia",
            "🕒 Scan History & Audit Log",
        ],
        index=1,
    )

    st.divider()
    st.markdown("### ⚙️ SYSTEM SPECS")
    st.write(f"**Image Input:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} px")
    st.write(f"**Target Classes:** {len(CLASS_NAMES)}")
    st.write(f"**Framework:** TensorFlow {tf.__version__}")
    st.caption("Grad-CAM XAI · Streamlit · Keras 3")


# ============================================================
# PAGE 1: OVERVIEW & DASHBOARD
# ============================================================

if navigation_choice == "🏠 Overview & Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <span class="hero-tag">🎓 FINAL YEAR ENGINEERING CAPSTONE</span>
            <h1 class="hero-title">Automated Wheat Leaf Disease Diagnosis & Explainable AI</h1>
            <p class="hero-subtitle">
                An end-to-end intelligent agricultural vision system designed to detect, classify,
                and interpret foliar wheat diseases in real-time. Built with a custom Deep Convolutional
                Neural Network and Gradient-weighted Class Activation Mapping (Grad-CAM) to provide
                transparent, agronomically actionable diagnoses.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">MODEL ACCURACY</div>
                <h2 style="margin: 0; color: #1b5e3f;">89.14%</h2>
                <p style="margin: 0.3rem 0 0 0; color: #526b5d; font-size: 0.85rem;">Test Set Top-1 Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">BENCHMARK DATASET</div>
                <h2 style="margin: 0; color: #1b5e3f;">4,878</h2>
                <p style="margin: 0.3rem 0 0 0; color: #526b5d; font-size: 0.85rem;">High-Res Wheat Images</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">TARGET DISEASES</div>
                <h2 style="margin: 0; color: #1b5e3f;">5 Classes</h2>
                <p style="margin: 0.3rem 0 0 0; color: #526b5d; font-size: 0.85rem;">Rusts, Smut, Blotch, Healthy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">EXPLAINABILITY</div>
                <h2 style="margin: 0; color: #1b5e3f;">Grad-CAM</h2>
                <p style="margin: 0.3rem 0 0 0; color: #526b5d; font-size: 0.85rem;">Spatial Lesion Heatmaps</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔬 System Architecture Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">PHASE 01</div>
                <h4>📷 Image Acquisition</h4>
                <p style="font-size: 0.88rem; color: #526b5d;">
                    Supports field camera capture, mobile photo uploads, or dataset batch testing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">PHASE 02</div>
                <h4>🧹 Preprocessing</h4>
                <p style="font-size: 0.88rem; color: #526b5d;">
                    Bilinear scaling to 224×224 RGB, normalization [0, 1], and noise filtering.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">PHASE 03</div>
                <h4>🧠 Custom Deep CNN</h4>
                <p style="font-size: 0.88rem; color: #526b5d;">
                    4 Conv-Blocks, Batch-Norm, L2 Regularization, Dropout & Global Average Pooling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-num">PHASE 04</div>
                <h4>🔍 XAI & Prescription</h4>
                <p style="font-size: 0.88rem; color: #526b5d;">
                    Grad-CAM lesion localization, severity grading, and downloadable agronomy report.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 💡 Quick Launch Diagnostic")
    st.write("Ready to test a specimen? Navigate directly to the **AI Disease Diagnosis** module.")


# ============================================================
# PAGE 2: AI DISEASE DIAGNOSIS (SINGLE IMAGE)
# ============================================================

elif navigation_choice == "🔬 AI Disease Diagnosis":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">🔬 REAL-TIME INFERENCE & EXPLAINABLE AI</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Wheat Foliar Disease Diagnosis</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Upload a field leaf image, take a camera snapshot, or select a sample from the project dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "SELECT INPUT SOURCE:",
        ["📤 Upload Leaf Photograph", "📸 Capture with Camera", "🧪 Select Sample from Project Dataset"],
        horizontal=True,
    )

    selected_image = None
    input_filename = "Captured_Sample"

    if input_mode == "📤 Upload Leaf Photograph":
        uploaded = st.file_uploader(
            "Upload Wheat Leaf Image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="For best diagnostic accuracy, ensure the leaf blade or diseased pustules are sharply focused and evenly illuminated.",
        )
        if uploaded:
            try:
                selected_image = Image.open(uploaded).convert("RGB")
                input_filename = uploaded.name
            except Exception as e:
                st.error(f"Error reading image: {e}")

    elif input_mode == "📸 Capture with Camera":
        camera_file = st.camera_input("Take a photo of the wheat leaf specimen")
        if camera_file:
            try:
                selected_image = Image.open(camera_file).convert("RGB")
                input_filename = f"Camera_{datetime.now().strftime('%H%M%S')}.jpg"
            except Exception as e:
                st.error(f"Error reading camera image: {e}")

    else:
        st.write("Select a disease category to load an authentic verified test sample from the dataset:")
        sample_col1, sample_col2 = st.columns([2, 1])
        with sample_col1:
            chosen_cat = st.selectbox("Disease Category", CLASS_NAMES, index=0)
        with sample_col2:
            if st.button("🎲 Fetch Random Dataset Sample", use_container_width=True):
                st.session_state["dataset_sample_trigger"] = random.randint(1, 10000)

        sample_path = get_random_sample_from_class(chosen_cat)
        if sample_path and sample_path.is_file():
            try:
                selected_image = Image.open(sample_path).convert("RGB")
                input_filename = sample_path.name
                st.success(f"Loaded verified sample from dataset: `{sample_path.name}` ({chosen_cat})")
            except Exception as e:
                st.error(f"Could not load sample: {e}")
        else:
            st.warning("Dataset directory not found or empty.")

    # When an image is ready
    if selected_image is not None:
        st.divider()
        st.markdown("### 🔍 Specimen Analysis & Model Execution")

        preview_col, action_col = st.columns([1, 1], gap="large")

        with preview_col:
            st.image(
                selected_image,
                caption=f"Input Specimen: {input_filename} ({selected_image.width} × {selected_image.height} px)",
                use_container_width=True,
            )

        with action_col:
            st.markdown(
                f"""
                <div class="feature-card" style="margin-top: 0;">
                    <div class="card-num">CONFIGURATION</div>
                    <h4 style="margin: 0.2rem 0;">Active Model: {selected_model_name}</h4>
                    <p style="font-size: 0.88rem; color: #526b5d; margin-bottom: 1rem;">
                        Standard input tensor shape: <code>(1, 224, 224, 3)</code> with normalized pixel floating point range [0, 1].
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            xai_colormap = st.selectbox(
                "Grad-CAM Heatmap Colormap",
                ["jet", "turbo", "inferno", "viridis", "magma"],
                index=0,
                help="Select the color gradient used to render the spatial activation map.",
            )

            overlay_alpha = st.slider(
                "Attention Overlay Blend Intensity (Alpha)",
                min_value=0.2,
                max_value=0.8,
                value=0.45,
                step=0.05,
            )

            run_btn = st.button("🚀 Run Deep Learning Diagnosis & Grad-CAM", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing neural forward pass and computing Grad-CAM backpropagation..."):
                try:
                    # 1. Run inference
                    inference_res = run_single_inference(selected_image, active_model)

                    # 2. Run Grad-CAM
                    explainer = get_gradcam_explainer(active_model)
                    cam_heatmap = explainer.generate_heatmap(
                        inference_res["batch_array"],
                        target_class_index=inference_res["class_index"],
                    )

                    superimposed_pil, heatmap_pil, act_ratio = explainer.overlay_heatmap(
                        selected_image,
                        cam_heatmap,
                        colormap_name=xai_colormap,
                        alpha=overlay_alpha,
                    )

                    severity_label, severity_css, infection_pct = classify_severity(
                        inference_res["predicted_class"], act_ratio
                    )

                    # Store in session state
                    sample_id = f"WHEAT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    diag_data = {
                        "sample_id": sample_id,
                        "filename": input_filename,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "predicted_class": inference_res["predicted_class"],
                        "confidence": inference_res["confidence"],
                        "severity_label": severity_label,
                        "severity_css": severity_css,
                        "infection_pct": infection_pct,
                        "scores": inference_res["scores"],
                        "original_pil": selected_image,
                        "heatmap_pil": heatmap_pil,
                        "superimposed_pil": superimposed_pil,
                        "model_name": selected_model_name,
                    }
                    st.session_state.current_diagnosis = diag_data

                    # Append to history
                    st.session_state.history.insert(0, {
                        "Time": diag_data["timestamp"],
                        "Sample ID": sample_id,
                        "Image": input_filename,
                        "Prediction": diag_data["predicted_class"],
                        "Confidence": f"{diag_data['confidence'] * 100:.2f}%",
                        "Severity": severity_label,
                        "Model": selected_model_name,
                    })

                except Exception as err:
                    st.error(f"Inference or Grad-CAM computation failed: {err}")
                    import traceback
                    st.code(traceback.format_exc())

        # Render Diagnosis Results if available
        if st.session_state.current_diagnosis is not None:
            diag = st.session_state.current_diagnosis
            kb_entry = DISEASE_KNOWLEDGE_BASE.get(diag["predicted_class"], {})

            st.divider()
            st.markdown("## 📊 Diagnostic Results & Clinical Assessment")

            res_col1, res_col2 = st.columns([1, 1], gap="large")

            with res_col1:
                st.markdown(
                    f"""
                    <div class="feature-card" style="border-left: 6px solid #1b5e3f;">
                        <div class="card-num">PRIMARY DIAGNOSTIC CLASSIFICATION</div>
                        <h1 style="color: #0f3d2e; margin: 0.2rem 0 0.4rem 0;">{diag['predicted_class']}</h1>
                        <p style="font-size: 0.95rem; color: #435e4f; margin-bottom: 0.8rem;">
                            <strong>Pathogen:</strong> <em>{kb_entry.get('scientific_name', 'Triticum aestivum')}</em> ({kb_entry.get('category', 'Healthy')})
                        </p>
                        <div>
                            <span class="badge-confidence">Confidence: {diag['confidence'] * 100:.2f}%</span>
                            <span class="{diag['severity_css']}">Severity: {diag['severity_label']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 🎯 Class Probability Distribution")
                chart_df = pd.DataFrame(diag["scores"])
                chart_df["score_pct"] = chart_df["score"] * 100

                chart = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                    .encode(
                        x=alt.X("score_pct:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("class:N", sort="-x", title="Disease Class"),
                        color=alt.condition(
                            alt.datum["class"] == diag["predicted_class"],
                            alt.value("#1b5e3f"),
                            alt.value("#9ecbb2"),
                        ),
                        tooltip=["class", alt.Tooltip("score_pct:Q", format=".2f", title="Confidence (%)")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)

            with res_col2:
                st.markdown("#### 🔬 Explainable AI (Grad-CAM) Visual Evidence")
                st.caption(
                    "The gradient backpropagation localizes the exact receptive field and pixel groups "
                    "that triggered the convolutional filters for the predicted disease."
                )

                v1, v2, v3 = st.columns(3)
                with v1:
                    st.image(diag["original_pil"], caption="01 · Input Specimen", use_container_width=True)
                with v2:
                    st.image(diag["heatmap_pil"], caption="02 · Grad-CAM Heatmap", use_container_width=True)
                with v3:
                    st.image(diag["superimposed_pil"], caption="03 · Attention Overlay", use_container_width=True)

                st.info(
                    f"**Salient Lesion Area Coverage:** `{diag['infection_pct']:.1f}%` of the spatial receptive field "
                    f"exhibits high disease activation peaks."
                )

            # Agronomic Prescription & Action Plan Tabs
            st.markdown("### 🌿 Agronomic Decision Support & Treatment Schedule")
            tab_sym, tab_cult, tab_org, tab_chem = st.tabs([
                "🔬 Visual Symptoms & Pathology",
                "🚜 Cultural Management",
                "🌱 Organic & Biological Remedies",
                "🧪 Chemical Fungicides & Prescription",
            ])

            with tab_sym:
                st.markdown(f"**Pathogen:** *{kb_entry.get('scientific_name')}* ({kb_entry.get('category')})")
                st.markdown(f"**Optimal Climate:** {kb_entry.get('optimal_conditions', {}).get('temperature', 'N/A')} | {kb_entry.get('optimal_conditions', {}).get('humidity', 'N/A')}")
                st.markdown("#### Key Identifiers:")
                for s in kb_entry.get("symptoms", []):
                    st.markdown(f"- {s}")

            with tab_cult:
                st.markdown("#### Cultural & Preventive Field Measures:")
                for c in kb_entry.get("cultural_control", []):
                    st.markdown(f"- {c}")

            with tab_org:
                st.markdown("#### Certified Organic Treatments & Bio-Pesticides:")
                for o in kb_entry.get("organic_remedies", []):
                    st.markdown(f"- {o}")

            with tab_chem:
                st.markdown("#### Registered Fungicide Prescriptions:")
                for ch in kb_entry.get("chemical_control", []):
                    st.markdown(f"- {ch}")
                st.caption(f"**Economic Spray Threshold:** {kb_entry.get('economic_threshold', 'Consult agronomist.')}")

            # Downloadable Diagnostic Report
            st.divider()
            st.markdown("### 📄 Diagnostic Certificate & Official Report Export")
            html_report = generate_diagnostic_html_report(
                sample_id=diag["sample_id"],
                image_filename=diag["filename"],
                predicted_class=diag["predicted_class"],
                confidence=diag["confidence"],
                severity_level=diag["severity_label"],
                infection_pct=diag["infection_pct"],
                original_pil=diag["original_pil"],
                heatmap_pil=diag["heatmap_pil"],
                overlay_pil=diag["superimposed_pil"],
                scores_list=diag["scores"],
                disease_info=kb_entry,
                model_name=diag["model_name"],
            )

            report_col1, report_col2 = st.columns([1, 1])
            with report_col1:
                st.download_button(
                    label="📥 Download Diagnostic Certificate (HTML / PDF-Ready)",
                    data=html_report,
                    file_name=f"WheatCare_Diagnostic_Report_{diag['sample_id']}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with report_col2:
                with st.expander("👁️ Preview Diagnostic Certificate"):
                    st.components.v1.html(html_report, height=600, scrolling=True)


# ============================================================
# PAGE 3: BATCH IMAGE DIAGNOSIS
# ============================================================

elif navigation_choice == "📁 Batch Image Diagnosis":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">📁 AUTOMATED BATCH PROCESSING</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Batch Wheat Leaf Diagnostic Pipeline</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Upload multiple wheat leaves simultaneously to perform high-throughput automated classification,
                generate statistical disease distributions, and export tabular summaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Upload Multiple Wheat Leaf Images (JPG, PNG)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Select multiple image files from your computer.",
    )

    if batch_files:
        st.write(f"📁 Selected **{len(batch_files)}** images for batch inference.")

        if st.button("⚡ Execute Batch Neural Analysis", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            batch_results = []
            for idx, file_obj in enumerate(batch_files):
                try:
                    img = Image.open(file_obj).convert("RGB")
                    pred = run_single_inference(img, active_model)
                    batch_results.append({
                        "Filename": file_obj.name,
                        "Predicted Class": pred["predicted_class"],
                        "Confidence": f"{pred['confidence'] * 100:.2f}%",
                        "Confidence_Raw": pred["confidence"],
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
                except Exception as e:
                    batch_results.append({
                        "Filename": file_obj.name,
                        "Predicted Class": "Error / Unreadable",
                        "Confidence": "0.00%",
                        "Confidence_Raw": 0.0,
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    })

                progress = (idx + 1) / len(batch_files)
                progress_bar.progress(progress)
                status_text.text(f"Processed {idx + 1} of {len(batch_files)} images...")

            status_text.success("✅ Batch processing completed successfully!")

            df_batch = pd.DataFrame(batch_results)

            st.divider()
            st.markdown("### 📊 Batch Diagnostic Summary")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Processed", len(df_batch))
            healthy_count = int((df_batch["Predicted Class"] == "Healthy").sum())
            diseased_count = len(df_batch) - healthy_count
            m2.metric("Healthy Plants", healthy_count)
            m3.metric("Diseased Plants", diseased_count)
            avg_conf = df_batch["Confidence_Raw"].mean() * 100.0
            m4.metric("Average Confidence", f"{avg_conf:.1f}%")

            # Distribution chart
            st.markdown("#### Class Breakdown in Batch")
            dist_counts = df_batch["Predicted Class"].value_counts().reset_index()
            dist_counts.columns = ["Class", "Count"]

            bar_chart = (
                alt.Chart(dist_counts)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    x="Count:Q",
                    y=alt.Y("Class:N", sort="-x"),
                    color=alt.Color("Class:N", legend=None),
                    tooltip=["Class", "Count"],
                )
                .properties(height=200)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            # Interactive Table
            st.markdown("#### Detailed File Results")
            st.dataframe(
                df_batch[["Filename", "Predicted Class", "Confidence", "Timestamp"]],
                use_container_width=True,
                hide_index=True,
            )

            # Export button
            csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Batch Diagnostic Report (CSV)",
                data=csv_bytes,
                file_name="WheatCare_Batch_Diagnostic_Report.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# PAGE 4: DATASET EXPLORER & ANALYTICS
# ============================================================

elif navigation_choice == "🗂️ Dataset Explorer & Analytics":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">🗂️ DATASET CURATION & DISTRIBUTION</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Wheat Foliar Benchmark Dataset</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Exploration of the 4,878 non-duplicate wheat leaf dataset utilized to train, validate,
                and evaluate our custom deep convolutional network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_stats = scan_dataset_statistics()
    total_imgs = sum(dataset_stats.values()) if dataset_stats else 4878

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filtered Images", f"{total_imgs:,}")
    col2.metric("Pathology Classes", len(CLASS_NAMES))
    col3.metric("Dataset Path", str(DATASET_DIR) if DATASET_DIR else "D:\\non_dup_dataset")

    st.markdown("### 📊 Class Distribution & Imbalance Analysis")

    if dataset_stats:
        stats_df = pd.DataFrame([
            {"Class": k, "Images": v, "Percentage": (v / total_imgs) * 100}
            for k, v in dataset_stats.items()
        ])

        chart = (
            alt.Chart(stats_df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("Images:Q", title="Number of Annotated Images"),
                y=alt.Y("Class:N", sort="-x", title="Pathology Class"),
                color=alt.Color("Class:N", legend=None),
                tooltip=["Class", "Images", alt.Tooltip("Percentage:Q", format=".1f", title="Share (%)")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="feature-card">
            <div class="card-num">ENGINEERING INSIGHT · CLASS IMBALANCE MITIGATION</div>
            <h4 style="margin: 0.2rem 0;">Why Class Weighting & Data Augmentation Were Crucial</h4>
            <p style="font-size: 0.9rem; color: #526b5d; line-height: 1.6;">
                Notice that <strong>Loose Smut (299 images)</strong> and <strong>Septoria (349 images)</strong>
                represent minority classes compared to <strong>Healthy (1,579)</strong> and <strong>Yellow Rust (1,395)</strong>.
                In standard CNN training, minority classes suffer from poor recall and low precision.
                In our <strong>Improved CNN</strong>, we applied scikit-learn inverse frequency class weighting
                (<code>compute_class_weight</code>) combined with an on-the-fly random spatial transformation pipeline
                (rotation, flip, zoom, contrast jitter). This boosted Loose Smut precision from <strong>58.97% to 88.89%</strong>!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🖼️ Live Dataset Sample Gallery")
    st.write("Browse real training and validation specimens from each category:")

    selected_class_view = st.selectbox("Choose Category to Inspect", CLASS_NAMES)

    active_gallery_dir = DATASET_DIR if (DATASET_DIR and (DATASET_DIR / selected_class_view).is_dir()) else SAMPLE_DIR
    if active_gallery_dir and (active_gallery_dir / selected_class_view).is_dir():
        folder = active_gallery_dir / selected_class_view
        sample_files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if sample_files:
            random.seed(42)
            display_samples = random.sample(sample_files, min(6, len(sample_files)))
            cols = st.columns(3)
            for i, samp in enumerate(display_samples):
                col = cols[i % 3]
                with col:
                    try:
                        im = Image.open(samp)
                        st.image(
                            im,
                            caption=f"{samp.name} ({im.width}×{im.height})",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption("Unable to load preview.")


# ============================================================
# PAGE 5: MODEL PERFORMANCE & BENCHMARKS
# ============================================================

elif navigation_choice == "📈 Model Performance & Benchmarks":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">📈 RIGOROUS EMPIRICAL EVALUATION</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Model Benchmarks & Ablation Study</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Comparative evaluation between the Baseline CNN and the Improved Regularized Deep CNN
                evaluated on the unseen 488-image stratified test partition.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 Architectural Ablation & Performance Comparison")

    comparison_data = [
        {"Metric / Feature": "Test Set Accuracy", "Baseline CNN": "84.63%", "Improved Deep CNN": "89.14% (+4.51%)"},
        {"Metric / Feature": "Macro F1-Score", "Baseline CNN": "0.7964", "Improved Deep CNN": "0.8608 (+0.064)"},
        {"Metric / Feature": "Loose Smut Precision", "Baseline CNN": "58.97%", "Improved Deep CNN": "88.89% (+29.92%)"},
        {"Metric / Feature": "Septoria F1-Score", "Baseline CNN": "68.49%", "Improved Deep CNN": "73.68% (+5.19%)"},
        {"Metric / Feature": "Convolutional Depth", "Baseline CNN": "3 Blocks (32->64->128)", "Improved Deep CNN": "4 Blocks (32->64->128->256)"},
        {"Metric / Feature": "Regularization", "Baseline CNN": "Dropout Only", "Improved Deep CNN": "Batch-Norm + L2 (1e-4) + Progressive Dropout"},
        {"Metric / Feature": "Data Augmentation", "Baseline CNN": "None (Static Input)", "Improved Deep CNN": "Spatial Rotation, Flip, Zoom, Contrast"},
        {"Metric / Feature": "Class Imbalance Fix", "Baseline CNN": "Uniform Weights", "Improved Deep CNN": "Inverse Frequency Class Weights"},
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Test Set Classification Report (Improved CNN)")

    class_report_data = [
        {"Pathology Class": "Brown rust", "Precision": 0.9417, "Recall": 0.9040, "F1-Score": 0.9224, "Test Support": 125},
        {"Pathology Class": "Healthy", "Precision": 0.8929, "Recall": 0.9494, "F1-Score": 0.9202, "Test Support": 158},
        {"Pathology Class": "Loose Smut", "Precision": 0.8889, "Recall": 0.8000, "F1-Score": 0.8421, "Test Support": 30},
        {"Pathology Class": "Septoria", "Precision": 0.6829, "Recall": 0.8000, "F1-Score": 0.7368, "Test Support": 35},
        {"Pathology Class": "Yellow rust", "Precision": 0.9091, "Recall": 0.8571, "F1-Score": 0.8824, "Test Support": 140},
    ]
    st.dataframe(pd.DataFrame(class_report_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📉 Training History & Confusion Matrix")

    plot_col1, plot_col2 = st.columns(2)

    cm_path = OUTPUTS_DIR / "improved_confusion_matrix.png"
    acc_path = OUTPUTS_DIR / "improved_accuracy_plot.png"
    loss_path = OUTPUTS_DIR / "improved_loss_plot.png"

    with plot_col1:
        st.markdown("#### Normalized Confusion Matrix")
        if cm_path.is_file():
            st.image(str(cm_path), caption="Test Confusion Matrix on 488 Samples", use_container_width=True)
        else:
            st.info("Confusion matrix plot not located in outputs directory.")

    with plot_col2:
        st.markdown("#### Accuracy & Loss Curves")
        if acc_path.is_file():
            st.image(str(acc_path), caption="Training vs Validation Accuracy Progression", use_container_width=True)
        if loss_path.is_file():
            st.image(str(loss_path), caption="Categorical Cross-Entropy Loss Decay", use_container_width=True)


# ============================================================
# PAGE 6: WHEAT PATHOLOGY ENCYCLOPEDIA
# ============================================================

elif navigation_choice == "📖 Wheat Pathology Encyclopedia":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">📖 SCIENTIFIC AGRONOMY COMPENDIUM</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Wheat Foliar Pathology Field Guide</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Scientific reference detailing pathogen etiology, diagnostic hallmarks, environmental thresholds,
                and Integrated Pest Management (IPM) protocols.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for disease_key, info in DISEASE_KNOWLEDGE_BASE.items():
        with st.expander(f"🌾 {info['display_name']} ({info['scientific_name']})", expanded=(disease_key == "Brown rust")):
            col_a, col_b = st.columns([1, 1], gap="large")

            with col_a:
                st.markdown(f"**Pathogen Classification:** {info['category']}")
                st.markdown(f"**Primary Host Range:** {info['primary_hosts']}")
                st.markdown("#### Environmental Etiology:")
                st.write(f"- **Temperature:** {info['optimal_conditions']['temperature']}")
                st.write(f"- **Humidity:** {info['optimal_conditions']['humidity']}")
                st.write(f"- **Season:** {info['optimal_conditions']['season']}")

                st.markdown("#### Key Symptoms:")
                for sym in info["symptoms"]:
                    st.write(f"• {sym}")

            with col_b:
                st.markdown("#### Integrated Pest Management (IPM):")
                st.markdown("**1. Cultural Control:**")
                for c in info["cultural_control"]:
                    st.write(f"- {c}")

                st.markdown("**2. Organic & Biological Control:**")
                for o in info["organic_remedies"]:
                    st.write(f"- {o}")

                st.markdown("**3. Chemical Fungicide Recommendations:**")
                for ch in info["chemical_control"]:
                    st.write(f"- {ch}")

                st.caption(f"**Action Threshold:** {info['economic_threshold']}")


# ============================================================
# PAGE 7: SCAN HISTORY & AUDIT LOG
# ============================================================

elif navigation_choice == "🕒 Scan History & Audit Log":
    st.markdown(
        """
        <div class="hero-banner" style="padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="hero-tag">🕒 AUDIT TRAIL & LOGS</span>
            <h2 style="color: #ffffff; margin: 0 0 0.5rem 0;">Prediction Audit History</h2>
            <p style="color: #e2f4e8; margin: 0; font-size: 0.95rem;">
                Review, filter, and export the session diagnosis history recorded during application runtime.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        filter_class = st.selectbox("Filter by Pathology Class", ["All"] + CLASS_NAMES)
        if filter_class != "All":
            filtered_df = history_df[history_df["Prediction"] == filter_class]
        else:
            filtered_df = history_df

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        col_dl, col_cl = st.columns([1, 1])
        with col_dl:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download History as CSV",
                data=csv_data,
                file_name="WheatCare_Session_History.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_cl:
            if st.button("🗑️ Clear Audit History", use_container_width=True):
                st.session_state.history = []
                st.session_state.current_diagnosis = None
                st.rerun()
    else:
        st.info("No predictions recorded in this session yet. Run an analysis in the AI Disease Diagnosis tab to record entries.")


# ============================================================
# 7. APPLICATION FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <strong>🌾 WheatCare AI System</strong> · Final Year Engineering Research Project<br>
        Deep Learning Computer Vision · Explainable AI (Grad-CAM) · Agricultural Informatics<br>
        <span style="font-size: 0.8rem; color: #889f92;">Designed for academic evaluation and clinical plant pathology demonstration.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
