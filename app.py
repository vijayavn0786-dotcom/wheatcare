"""
WheatCare AI - Intelligent Wheat Leaf Disease Diagnosis & Agronomic Decision Support System
A Complete Final Year Engineering Showcase in Deep Learning, Computer Vision & Agritech.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Headless configuration
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go

# Modular system components
from gradcam import GradCAMExplainer
from disease_kb import DISEASE_KNOWLEDGE_BASE
from report_generator import generate_diagnostic_html_report


# ============================================================
# 1. PAGE & APPLICATION CONFIGURATION
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
SAMPLE_DIR = BASE_DIR / "sample_images"

IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 2. CUSTOM CSS STYLING & AGRITECH DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

    :root {
        --primary-deep: #0f3d2e;
        --primary: #1b5e3f;
        --primary-emerald: #2ecc71;
        --surface: #ffffff;
        --surface-subtle: #f8faf8;
        --border-color: #dce7df;
        --text-dark: #14281f;
        --text-muted: #526b5d;
        --danger-red: #c0392b;
        --warning-orange: #d35400;
        --purple-septoria: #8e44ad;
        --dark-smut: #2c3e50;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #f5f8f6;
        color: var(--text-dark);
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--primary-deep) !important;
        font-weight: 700 !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0e3728 0%, #175238 50%, #206d4b 100%);
        color: #ffffff;
        padding: 2.0rem 2.4rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(15, 61, 46, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
        color: #ffffff !important;
    }
    .hero-sub {
        font-size: 1.05rem;
        opacity: 0.90;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 0.8rem;
        margin-right: 8px;
    }
    .badge-emerald {
        background: rgba(46, 204, 113, 0.2);
        color: #a3f7c4;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .badge-white {
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Modern Card */
    .modern-card {
        background-color: var(--surface);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.2rem;
    }

    /* Status Banners */
    .status-banner {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.2rem;
    }
    .status-healthy {
        background: #e8f8f0;
        border: 1.5px solid #2ecc71;
        color: #1e824c;
    }
    .status-danger {
        background: #fdeeee;
        border: 1.5px solid #e74c3c;
        color: #962d22;
    }
    .status-warning {
        background: #fef5ea;
        border: 1.5px solid #e67e22;
        color: #b95d09;
    }
    .status-septoria {
        background: #f5eef8;
        border: 1.5px solid #9b59b6;
        color: #6c3483;
    }
    .status-smut {
        background: #ebedef;
        border: 1.5px solid #34495e;
        color: #1b2631;
    }

    .status-icon {
        font-size: 2.0rem;
    }
    .status-title {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .status-subtitle {
        font-size: 0.88rem;
        opacity: 0.85;
        margin-top: 3px;
    }

    /* Metric Scorecard */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .metric-pill {
        background: #f8faf8;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.85rem 1.0rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: var(--text-muted);
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--primary-deep);
        margin-top: 2px;
    }

    /* File uploader styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #a2c7b1;
        border-radius: 14px;
        padding: 10px;
        background: #f9fbf9;
    }

    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        background-color: #ffffff;
        border: 1px solid #dce7df;
        border-radius: 10px 10px 0 0;
        color: #526b5d;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: #ffffff !important;
        border-color: var(--primary) !important;
    }

    /* Quick Tip Box */
    .tip-box {
        background: #eef7f2;
        border-left: 4px solid var(--primary);
        padding: 0.9rem 1.2rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        margin-bottom: 1rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL & DATA CACHING
# ============================================================

@st.cache_resource(show_spinner="Loading trained Deep Neural Models...")
def load_all_models():
    """Caches and returns available Keras models, TFLite interpreter, and class names."""
    models = {}
    class_names = [
        "Brown rust",
        "Healthy",
        "Loose Smut",
        "Septoria",
        "Yellow rust",
    ]

    # Load class names from json if present
    if CLASS_NAMES_PATH.is_file():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception:
            pass

    # 1. Improved CNN
    if MODEL_IMPROVED_PATH.is_file():
        try:
            models["Improved CNN (89.14% Accuracy) [Recommended]"] = tf.keras.models.load_model(
                MODEL_IMPROVED_PATH
            )
        except Exception as e:
            st.warning(f"Failed to load Improved CNN: {e}")

    # 2. Baseline CNN
    if MODEL_BASELINE_PATH.is_file():
        try:
            models["Baseline CNN (84.63% Accuracy)"] = tf.keras.models.load_model(
                MODEL_BASELINE_PATH
            )
        except Exception as e:
            st.warning(f"Failed to load Baseline CNN: {e}")

    return models, class_names


@st.cache_resource
def get_gradcam_explainer(_model):
    """Caches Grad-CAM explainer for selected model."""
    try:
        return GradCAMExplainer(_model)
    except Exception:
        return None


# ============================================================
# 4. PREPROCESSING & INFERENCE FUNCTIONS
# ============================================================

def preprocess_image(pil_image):
    """
    Standardizes image to 224x224 RGB, normalized to [0, 1].
    Returns (preprocessed_array, display_image).
    """
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    resized_img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    batch_array = np.expand_dims(img_array, axis=0)
    return batch_array, img


def run_prediction(model, preprocessed_array, class_names):
    """Runs forward pass, returns top prediction, confidence, probabilities and latency."""
    start_time = time.time()
    predictions = model.predict(preprocessed_array, verbose=0)
    latency_ms = int((time.time() - start_time) * 1000)

    probabilities = predictions[0]
    top_idx = int(np.argmax(probabilities))
    top_class = class_names[top_idx]
    top_confidence = float(probabilities[top_idx]) * 100.0

    return top_class, top_confidence, probabilities, latency_ms


def plot_probability_distribution(class_names, probabilities, predicted_class):
    """Creates a stylized Plotly horizontal bar chart highlighting the winner class."""
    palette = {
        "Healthy": "#2ecc71",
        "Yellow rust": "#e74c3c",
        "Brown rust": "#e67e22",
        "Septoria": "#9b59b6",
        "Loose Smut": "#34495e",
    }

    colors = []
    for cls in class_names:
        if cls == predicted_class:
            colors.append(palette.get(cls, "#1b5e3f"))
        else:
            colors.append("#d0dad4")

    # Sort ascending for natural top-down horizontal chart
    sorted_data = sorted(
        zip(class_names, probabilities, colors),
        key=lambda x: x[1],
        reverse=False,
    )
    y_classes = [item[0] for item in sorted_data]
    x_probs = [item[1] * 100.0 for item in sorted_data]
    bar_colors = [item[2] for item in sorted_data]

    fig = go.Figure(
        go.Bar(
            x=x_probs,
            y=y_classes,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#0f3d2e", width=1),
            ),
            text=[f"<b>{val:.1f}%</b>" for val in x_probs],
            textposition="auto",
            textfont=dict(color="#14281f", size=12, family="Plus Jakarta Sans"),
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Pathology Probability Breakdown</b>",
            font=dict(size=14, color="#0f3d2e", family="Space Grotesk"),
        ),
        xaxis=dict(
            title="Class Probability (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor="#e9f0eb",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13, color="#14281f", family="Plus Jakarta Sans"),
        ),
        margin=dict(l=10, r=20, t=35, b=25),
        height=270,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


# ============================================================
# 5. SIDEBAR: METADATA, DEMO SELECTOR & CONTROLS
# ============================================================

def render_sidebar(models):
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="font-size:2rem;">🌾</span>
                <div>
                    <h2 style="margin:0; font-size:1.3rem; color:#0f3d2e;">WheatCare AI</h2>
                    <span style="font-size:0.75rem; color:#526b5d; font-weight:600;">PRECISION AGRITECH DECISION ENGINE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="tip-box">
                <b>Judge Evaluation Mode:</b><br>
                Test real-time diagnosis instantly by picking any quick demo sample below, or upload custom field foliage photos.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Try Quick Demos Section
        st.markdown("### ⚡ Try Quick Demos")
        demo_options = [
            "-- None (Upload Leaf) --",
            "🟢 Healthy Wheat (Control Leaf)",
            "🟡 Yellow Rust (Stripe Rust)",
            "🟠 Brown Rust (Leaf Rust)",
            "🟣 Septoria Leaf Blotch",
            "⚫ Loose Smut (Blackened Spike)",
        ]
        selected_demo_label = st.selectbox(
            "Select Benchmark Sample:",
            demo_options,
            index=0,
            help="Select one of the 5 verified wheat pathology samples to diagnose in 1 click.",
        )

        demo_mapping = {
            "🟢 Healthy Wheat (Control Leaf)": SAMPLE_DIR / "Healthy" / "0LBIWV.jpg",
            "🟡 Yellow Rust (Stripe Rust)": SAMPLE_DIR / "Yellow rust" / "YellowRust1003.png",
            "🟠 Brown Rust (Leaf Rust)": SAMPLE_DIR / "Brown rust" / "BrownRust1018.png",
            "🟣 Septoria Leaf Blotch)": SAMPLE_DIR / "Septoria" / "Septoria1011.png",
            "⚫ Loose Smut (Blackened Spike)": SAMPLE_DIR / "Loose Smut" / "0021.jpg",
        }

        # 2. Model Selection
        st.markdown("### 🧠 Neural Model")
        model_names = list(models.keys())
        selected_model_name = st.selectbox(
            "Active Architecture:",
            model_names,
            index=0,
            help="Select the deep learning model pipeline used for visual inference.",
        )

        # 3. Model & Project Specifications Card
        st.markdown("### ⚙️ System Specifications")
        st.markdown(
            """
            <div style="background:#ffffff; border:1px solid #dce7df; border-radius:12px; padding:12px; font-size:0.8rem; line-height:1.6;">
                <b>Architecture:</b> Custom Deep CNN<br>
                <b>Training Accuracy:</b> 89.14% (Macro F1 86.1%)<br>
                <b>Input Tensor:</b> 224 x 224 x 3 (Bilinear RGB)<br>
                <b>Classes (5):</b> Healthy, Brown Rust, Yellow Rust, Septoria, Loose Smut<br>
                <b>Decision Protocols:</b> ICAR & FAO Guidelines
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Grad-CAM Explainable AI Toggle
        st.markdown("### 🔬 Explainability")
        enable_gradcam = st.checkbox(
            "Enable Grad-CAM Visual Heatmap",
            value=True,
            help="Computes Gradient-weighted Class Activation Mapping to display the exact visual features triggering the prediction.",
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem; color:#526b5d; text-align:center;'>Final Year Engineering Showcase • Antigravity AI</div>",
            unsafe_allow_html=True,
        )

    return selected_demo_label, selected_model_name, enable_gradcam


# ============================================================
# 6. MAIN APPLICATION EXECUTION
# ============================================================

def main():
    models, class_names = load_all_models()
    if not models:
        st.error("No trained models found in the models directory. Please verify project weights.")
        st.stop()

    # Render Sidebar
    selected_demo_label, selected_model_name, enable_gradcam = render_sidebar(models)
    active_model = models[selected_model_name]

    # Top Hero Header
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🌾 WheatCare AI: Intelligent Crop Pathology Showcase</div>
            <div class="hero-sub">
                Real-time on-device & cloud wheat leaf diagnostic system powered by Convolutional Neural Networks (CNNs).
                Diagnoses fungal diseases in milliseconds and generates verified agronomic treatment protocols.
            </div>
            <div>
                <span class="badge-pill badge-emerald">⚡ 89.14% Test Accuracy</span>
                <span class="badge-pill badge-white">🔬 Explainable AI (Grad-CAM)</span>
                <span class="badge-pill badge-white">🌱 ICAR/FAO Agronomic Advisory</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-Column Workspace
    col_left, col_right = st.columns([1, 1], gap="large")

    active_pil_image = None
    image_source_info = ""

    with col_left:
        st.markdown("### 📸 Specimen Input & Visual Analysis")

        uploaded_file = st.file_uploader(
            "Upload Wheat Leaf Specimen:",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Upload a clear photograph of a single wheat leaf or spikelet.",
        )

        # Priority: 1. Uploaded file, 2. Quick Demo selected
        if uploaded_file is not None:
            try:
                active_pil_image = Image.open(uploaded_file)
                image_source_info = f"Uploaded Specimen: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
        elif selected_demo_label != "-- None (Upload Leaf) --":
            # Find demo file
            demo_keyword = selected_demo_label.split()[1] # e.g. Healthy, Yellow, Brown, Septoria, Loose
            for cls_folder in SAMPLE_DIR.glob("*"):
                if demo_keyword.lower() in cls_folder.name.lower():
                    sample_files = list(cls_folder.glob("*.*"))
                    if sample_files:
                        active_pil_image = Image.open(sample_files[0])
                        image_source_info = f"Benchmark Sample: {cls_folder.name} ({sample_files[0].name})"
                        break

        # Display Image Viewport
        if active_pil_image is not None:
            st.image(
                active_pil_image,
                caption=image_source_info,
                use_container_width=True,
            )

            # Image Metadata Scoreboard
            w, h = active_pil_image.size
            aspect = round(w / h, 2)
            mode = active_pil_image.mode

            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-pill">
                        <div class="metric-label">Resolution</div>
                        <div class="metric-val" style="font-size:1.05rem;">{w} × {h} px</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-label">Channel Format</div>
                        <div class="metric-val" style="font-size:1.05rem;">{mode}</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-label">Aspect Ratio</div>
                        <div class="metric-val" style="font-size:1.05rem;">{aspect}:1</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Preprocessing Expander
            with st.expander("🔍 View Preprocessed 224x224 Normalized Tensor", expanded=False):
                batch_arr, display_img = preprocess_image(active_pil_image)
                st.image(
                    display_img.resize((224, 224)),
                    caption="Resampled 224x224 RGB Tensor Input (Bilinear)",
                    width=224,
                )
                st.code(
                    f"Tensor Shape: {batch_arr.shape}\n"
                    f"Min Pixel Value: {batch_arr.min():.3f} | Max Pixel Value: {batch_arr.max():.3f}\n"
                    f"Dtype: {batch_arr.dtype}",
                    language="text",
                )
        else:
            st.info("👈 Please select a quick demo from the left sidebar or upload a wheat foliage image above.")

    with col_right:
        st.markdown("### 🔬 Neural Classification & Probabilities")

        if active_pil_image is not None:
            # Execute Model Prediction
            batch_arr, _ = preprocess_image(active_pil_image)
            top_class, top_confidence, probabilities, latency_ms = run_prediction(
                active_model, batch_arr, class_names
            )

            # Status Banner Mapping
            banner_html = ""
            if top_class == "Healthy":
                banner_html = f"""
                <div class="status-banner status-healthy">
                    <div class="status-icon">🟢</div>
                    <div>
                        <div class="status-title">Healthy Wheat Leaf</div>
                        <div class="status-subtitle">Optimal Physiological State — No Pathogens Detected ({top_confidence:.1f}% Confidence)</div>
                    </div>
                </div>
                """
            elif top_class == "Yellow rust":
                banner_html = f"""
                <div class="status-banner status-danger">
                    <div class="status-icon">🔴</div>
                    <div>
                        <div class="status-title">Yellow Rust (Stripe Rust) Detected</div>
                        <div class="status-subtitle">Critical Airborne Risk (Puccinia striiformis) — {top_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """
            elif top_class == "Brown rust":
                banner_html = f"""
                <div class="status-banner status-warning">
                    <div class="status-icon">🟠</div>
                    <div>
                        <div class="status-title">Brown Rust (Leaf Rust) Detected</div>
                        <div class="status-subtitle">High Severity (Puccinia triticina) — {top_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """
            elif top_class == "Septoria":
                banner_html = f"""
                <div class="status-banner status-septoria">
                    <div class="status-icon">🟣</div>
                    <div>
                        <div class="status-title">Septoria Leaf Blotch Detected</div>
                        <div class="status-subtitle">Moderate to Severe Necrosis (Zymoseptoria tritici) — {top_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """
            else:
                banner_html = f"""
                <div class="status-banner status-smut">
                    <div class="status-icon">⚫</div>
                    <div>
                        <div class="status-title">Loose Smut Detected</div>
                        <div class="status-subtitle">Systemic Seed-Borne Smut (Ustilago tritici) — {top_confidence:.1f}% Confidence</div>
                    </div>
                </div>
                """

            st.markdown(banner_html, unsafe_allow_html=True)

            # Scoreboard Metrics
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-pill">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-val" style="font-size:1.15rem;">{top_class}</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-val" style="color:#1b5e3f;">{top_confidence:.1f}%</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-label">Latency</div>
                        <div class="metric-val" style="font-size:1.15rem;">⚡ {latency_ms} ms</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Plotly Horizontal Probability Breakdown
            fig = plot_probability_distribution(class_names, probabilities, top_class)
            st.plotly_chart(fig, use_container_width=True)

            # Explainable AI: Grad-CAM
            if enable_gradcam:
                with st.spinner("Generating Grad-CAM visual heatmap..."):
                    explainer = get_gradcam_explainer(active_model)
                    if explainer:
                        try:
                            heatmap = explainer.generate_heatmap(batch_arr)
                            overlay = explainer.overlay_heatmap(active_pil_image, heatmap)
                            st.markdown("#### 🔬 Explainable AI: Grad-CAM Visual Attention")
                            st.image(
                                overlay,
                                caption="Salient visual features and pathogen focus regions identified by Conv2D layers",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.caption(f"Grad-CAM visualization skipped: {e}")

            # Download Diagnostic Report Button
            kb_data = DISEASE_KNOWLEDGE_BASE.get(top_class, {})
            report_html = generate_diagnostic_html_report(
                top_class,
                top_confidence,
                probabilities,
                class_names,
                latency_ms,
                kb_data,
            )
            st.download_button(
                label="📥 Download Official Agronomic Diagnostic Report (HTML)",
                data=report_html,
                file_name=f"WheatCare_Diagnostic_Report_{top_class.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div class="modern-card" style="text-align:center; padding:3rem 1.5rem;">
                    <span style="font-size:3rem;">🌱</span>
                    <h4 style="margin-top:12px;">Awaiting Leaf Specimen</h4>
                    <p style="color:#526b5d; font-size:0.9rem;">
                        Upload a photo or select any benchmark demo from the left sidebar to trigger instant deep neural analysis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ============================================================
    # 7. BOTTOM SECTION: COMPREHENSIVE AGRONOMIC ADVISORY TABS
    # ============================================================

    st.markdown("---")
    st.markdown("### 🌾 Comprehensive Agronomic Advisory & Crop Science Protocols")

    # Determine active disease info for tabs
    active_disease = top_class if active_pil_image is not None else "Yellow rust"
    kb = DISEASE_KNOWLEDGE_BASE.get(active_disease, DISEASE_KNOWLEDGE_BASE["Yellow rust"])

    tab1, tab2, tab3, tab4 = st.tabs([
        f"📋 Symptoms & Pathology Profile ({active_disease})",
        "💊 Fungicide & Chemical Dosage (ICAR/FAO)",
        "🌿 Organic & Cultural Management",
        "📊 Model Architecture & Evaluation Metrics",
    ])

    with tab1:
        st.markdown(f"#### Disease Profile: {kb.get('display_name', active_disease)}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"""
                <div class="modern-card">
                    <b>Scientific Taxonomy:</b> <i>{kb.get('scientific_name', 'N/A')}</i><br>
                    <b>Pathogen Category:</b> {kb.get('category', 'N/A')}<br>
                    <b>Primary Plant Hosts:</b> {kb.get('primary_hosts', 'Wheat')}<br>
                    <b>Economic Damage Threshold:</b> {kb.get('economic_threshold', 'Consult local agronomist.')}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("##### 🔬 Microclimatic Incubation Thresholds")
            cond = kb.get("optimal_conditions", {})
            st.markdown(
                f"""
                - **Temperature:** {cond.get('temperature', 'N/A')}
                - **Humidity & Wetness:** {cond.get('humidity', 'N/A')}
                - **Seasonal Vulnerability:** {cond.get('season', 'N/A')}
                """
            )
        with c2:
            st.markdown("##### 🔍 Visible Foliar Symptoms")
            for sym in kb.get("symptoms", []):
                st.markdown(f"- {sym}")

            st.markdown("##### ⚠️ Severity Classification")
            grades = kb.get("severity_grade", {})
            st.markdown(
                f"""
                - **Mild:** {grades.get('mild', 'N/A')}
                - **Moderate:** {grades.get('moderate', 'N/A')}
                - **Severe:** {grades.get('severe', 'N/A')}
                """
            )

    with tab2:
        st.markdown(f"#### 💊 Approved Chemical Fungicide Prescriptions (ICAR & FAO Standards)")
        st.markdown(
            """
            <div class="tip-box">
                <b>Standard Spray Guidance:</b> Apply using a calibrated knapsack sprayer with hollow-cone nozzles. 
                Ensure a spray volume of 400–500 Liters of water per hectare for complete canopy coverage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        chemicals = kb.get("chemical_control", [])
        if chemicals:
            for idx, chem in enumerate(chemicals, 1):
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border-left:4px solid #c0392b; padding:12px 16px; border-radius:0 12px 12px 0; margin-bottom:10px; border-top:1px solid #dce7df; border-right:1px solid #dce7df; border-bottom:1px solid #dce7df;">
                        <b style="color:#c0392b;">Prescription #{idx}:</b> {chem}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No chemical intervention required for this healthy crop specimen.")

        st.warning(
            "⚠️ **Pre-Harvest Interval (PHI) Notice:** Adhere strictly to the statutory 30-day PHI between fungicide spraying and grain harvest. Always wear personal protective equipment (PPE)."
        )

    with tab3:
        st.markdown(f"#### 🌿 Organic Remedies & Preventive Cultural Practices")
        col_org1, col_org2 = st.columns(2)
        with col_org1:
            st.markdown("##### 🌱 Biological & Organic Formulations")
            for bio in kb.get("organic_remedies", []):
                st.markdown(f"- {bio}")

        with col_org2:
            st.markdown("##### 🚜 Agronomic Cultural Management")
            for cult in kb.get("cultural_control", []):
                st.markdown(f"- {cult}")

        st.markdown("##### 🌾 High-Yielding Resistant Cultivars")
        st.info(
            "Recommended Rust & Blight Tolerant Wheat Varieties (ICAR-IIWBR): **HD-3086 (Pusa Gautami)**, **DBW-187 (Karan Vandana)**, **DBW-303 (Karan Vaishnavi)**, **PBW-550**, and **WH-1105**."
        )

    with tab4:
        st.markdown("#### 📊 Model Architecture & Test Evaluation Benchmarks")
        m_col1, m_col2 = st.columns([1, 1])

        with m_col1:
            st.markdown("##### 📈 Test Classification Report (488 Validation Images)")
            perf_df = pd.DataFrame(
                [
                    {"Class": "Brown rust", "Precision": 0.9417, "Recall": 0.9040, "F1-Score": 0.9224, "Test Images": 125},
                    {"Class": "Healthy", "Precision": 0.8929, "Recall": 0.9494, "F1-Score": 0.9202, "Test Images": 158},
                    {"Class": "Loose Smut", "Precision": 0.8889, "Recall": 0.8000, "F1-Score": 0.8421, "Test Images": 30},
                    {"Class": "Septoria", "Precision": 0.6829, "Recall": 0.8000, "F1-Score": 0.7368, "Test Images": 35},
                    {"Class": "Yellow rust", "Precision": 0.9091, "Recall": 0.8571, "F1-Score": 0.8824, "Test Images": 140},
                ]
            )
            st.dataframe(
                perf_df.style.format({
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                """
                <div style="background:#f8faf8; border:1px solid #dce7df; border-radius:10px; padding:10px; font-size:0.85rem;">
                    <b>Overall Model Accuracy:</b> 89.14%<br>
                    <b>Macro Average F1:</b> 86.08%<br>
                    <b>Weighted Average F1:</b> 89.20%
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            st.markdown("##### 🎯 Confusion Matrix")
            cm_img_path = OUTPUTS_DIR / "improved_confusion_matrix.png"
            if cm_img_path.is_file():
                st.image(str(cm_img_path), caption="Normalized Confusion Matrix (Test Set)", use_container_width=True)
            else:
                st.caption("Confusion matrix visualization will appear here.")


if __name__ == "__main__":
    main()
