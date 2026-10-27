---
title: WheatCare - Intelligent Wheat Disease Diagnosis & Explainable AI
emoji: 🌾
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# 🌾 WheatCare: Intelligent Wheat Foliar Disease Diagnosis & Explainable AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-red.svg?logo=streamlit)](https://streamlit.io/)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-green.svg)](https://arxiv.org/abs/1610.02391)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-89.14%25-brightgreen.svg)]()

> **Final Year Engineering Capstone Project**  
> An end-to-end Computer Vision & Explainable Deep Learning platform designed to detect, classify, and interpret wheat leaf diseases in real-time.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Target Pathology Classes](#-target-pathology-classes)
3. [Deep CNN Architecture](#-deep-cnn-architecture)
4. [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
5. [Model Benchmarks & Results](#-model-benchmarks--results)
6. [Quickstart & Deployment](#-quickstart--deployment)
   - [A. 1-Click Launch on Windows](#a-1-click-launch-on-windows)
   - [B. Publish to GitHub](#b-publish-to-github)
   - [C. Streamlit Community Cloud (Free Public URL)](#c-streamlit-community-cloud-free-public-url)
   - [D. Hugging Face Spaces (Free Cloud Hosting)](#d-hugging-face-spaces-free-cloud-hosting)
   - [E. Docker Container](#e-docker-container)
7. [Repository Structure](#-repository-structure)

---

## 🌟 Project Overview

Wheat (*Triticum aestivum*) is a vital cereal crop feeding over 2.5 billion people. Foliar fungal diseases devastate crop yield and quality when not diagnosed early. **WheatCare** bridges modern computer vision and agronomy:
- **Instant Diagnostic Classification:** Categorizes wheat leaves into 5 classes with softmax confidence scores.
- **Explainable AI (Grad-CAM):** Visualizes the exact receptive field and convolutional features that triggered the model.
- **Infection Severity Grading:** Analyzes lesion area percentage to classify infection severity (Mild, Moderate, Severe).
- **Agronomic Action Plans:** Provides cultural methods, organic bio-remedies, and registered fungicide active ingredients.
- **Report Generation:** Generates downloadable and printable clinical diagnostic certificates in HTML/PDF format.

---

## 🍂 Target Pathology Classes

| Pathology Class | Pathogen / Organism | Symptoms & Hallmarks |
| :--- | :--- | :--- |
| **Brown Rust** | *Puccinia triticina* | Scattered circular to oval reddish-brown powdery pustules on upper leaf blades. |
| **Yellow Rust** | *Puccinia striiformis* | Bright lemon-yellow pustules arranged in parallel linear stripes along veins. |
| **Loose Smut** | *Ustilago tritici* | Olive-black powdery teliospore masses completely replacing grain spikelets. |
| **Septoria Blotch** | *Zymoseptoria tritici* | Irregular tan-brown necrotic blotches containing tiny black fruiting bodies (pycnidia). |
| **Healthy Foliage** | *Triticum aestivum* | Vigorous green leaves devoid of lesions, chlorosis, or fungal pustules. |

---

## 🧠 Deep CNN Architecture

Our custom deep convolutional neural network features 4 convolutional blocks engineered to eliminate overfitting:
- **Block 1:** Conv2D (32 filters, 3×3) $\to$ BatchNorm $\to$ Conv2D (32 filters) $\to$ MaxPool (2×2) $\to$ Dropout (0.15)
- **Block 2:** Conv2D (64 filters, 3×3) $\to$ BatchNorm $\to$ Conv2D (64 filters) $\to$ MaxPool (2×2) $\to$ Dropout (0.20)
- **Block 3:** Conv2D (128 filters, 3×3) $\to$ BatchNorm $\to$ Conv2D (128 filters) $\to$ MaxPool (2×2) $\to$ Dropout (0.25)
- **Block 4:** Conv2D (256 filters, 3×3) $\to$ BatchNorm $\to$ Conv2D (256 filters) $\to$ MaxPool (2×2) $\to$ Dropout (0.30)
- **Head:** GlobalAveragePooling2D $\to$ Dense (128 units, ReLU, L2=1e-4) $\to$ BatchNorm $\to$ Dropout (0.35) $\to$ Dense (5 units, Softmax)

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM extracts gradients of the winning class score $y^c$ with respect to the feature activation maps $A^k$ of `conv2d_7`:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i, j}}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

This guarantees clinical accountability by confirming the network focuses on real lesions instead of background noise.

---

## 📊 Model Benchmarks & Results

Evaluated on an independent, unseen test partition of **488 stratified samples**:

| Metric | Baseline CNN | Improved Deep CNN | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **84.63%** | **89.14%** | **+4.51%** |
| **Macro Average F1** | 0.7964 | **0.8608** | **+0.064** |
| **Loose Smut Precision** | 58.97% | **88.89%** | **+29.92%** |
| **Septoria F1-Score** | 68.49% | **73.68%** | **+5.19%** |

---

## 🚀 Quickstart & Deployment

### A. 1-Click Launch on Windows
Double-click `run_app.bat` inside `D:\wheatcare\`:
```powershell
D:\wheatcare\run_app.bat
```
App opens at: `http://localhost:8501`

---

### B. Publish to GitHub
```bash
cd "D:\wheatcare"
git remote add origin https://github.com/vijayavn0786/wheatcare.git
git push -u origin main
```
*(Or in **GitHub Desktop**: `File` -> `Add Local Repository` -> Select `D:\wheatcare` -> Click `Publish repository`)*

---

### C. Streamlit Community Cloud (Free Public URL)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New App"** $\to$ Select repository `vijayavn0786/wheatcare` $\to$ Main file: `app.py`.
3. Click **"Deploy"**! You will receive a live public URL (e.g., `https://wheatcare.streamlit.app`) to share with professors and examiners.

---

### D. Hugging Face Spaces (Free Cloud Hosting)
1. Create a free account at [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name it `wheatcare`, choose **Streamlit**, and select the **Free CPU** tier.
3. Push this repository to the Hugging Face remote:
   ```bash
   git remote add space https://huggingface.co/spaces/vijayavn0786/wheatcare
   git push space main
   ```
4. Hugging Face will automatically build and host the application!

---

### E. Docker Container
Build and run anywhere with Docker:
```bash
docker build -t wheatcare .
docker run -p 8501:8501 wheatcare
```
Visit `http://localhost:8501` in your browser.

---

## 📂 Repository Structure

```
├── app.py                      # Master Streamlit web application
├── gradcam.py                  # Grad-CAM XAI spatial visualizer
├── disease_kb.py               # Scientific agronomy compendium
├── report_generator.py         # Printable/downloadable diagnostic reports
├── run_app.bat                 # 1-Click Windows desktop launcher
├── Dockerfile                  # Containerized deployment spec
├── requirements.txt            # Python dependencies
├── PROJECT_DOCUMENTATION.md    # Full project report & Viva Voce guide
├── models/
│   ├── wheat_leaf_disease_cnn_improved.keras  # 89.14% Accuracy model
│   ├── wheat_leaf_disease_cnn.keras           # Baseline reference model
│   └── class_names.json                       # 5 target classes mapping
├── sample_images/              # Bundled samples for instant demo testing
└── outputs/                    # Confusion matrix & performance graphs
```

---

## 📜 Academic Declaration & License
Developed as a **Final Year Engineering Capstone Project**. Released under the [MIT License](LICENSE).
