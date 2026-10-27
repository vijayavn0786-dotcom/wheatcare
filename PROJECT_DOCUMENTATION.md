# WheatCare AI: Intelligent Wheat Foliar Disease Diagnosis & Explainable AI

**Final Year Engineering Capstone Project**  
*Domain:* Computer Vision, Deep Learning, Agricultural Informatics, Explainable AI (XAI)  
*Tech Stack:* Python 3.12, TensorFlow 2.16+ / Keras 3, Streamlit, NumPy, Scikit-Learn, Matplotlib, PIL  

---

## 1. Executive Summary & Abstract

Wheat (*Triticum aestivum*) is a cornerstone of global food security, accounting for over 20% of all human caloric intake. However, foliar diseases such as **Brown Rust (*Puccinia triticina*)**, **Yellow Rust (*Puccinia striiformis*)**, **Septoria Leaf Blotch (*Zymoseptoria tritici*)**, and **Loose Smut (*Ustilago tritici*)** cause billions of dollars in global annual yield losses if not detected early.

**WheatCare AI** is an end-to-end intelligent agricultural vision system developed to provide automated, real-time diagnostic classification and spatial lesion interpretation directly from leaf photographs. The project incorporates:
1. **A Custom 9-Layer Deep Convolutional Neural Network (CNN)** engineered with Batch Normalization, L2 kernel regularization, progressive dropout, and data augmentation to mitigate overfitting.
2. **Explainable AI (Grad-CAM)**: Generates spatial activation heatmaps to interpret convolutional filter activations, ensuring clinical transparency and visual verification for agronomists and farmers.
3. **Automated Severity Grading**: Computes lesion surface area ratios to grade infections as Mild, Moderate, or Severe.
4. **Agronomic Clinical Decision Support**: Supplies immediate containment steps, organic remedies, and active fungicide chemical prescriptions.
5. **Interactive Web Application**: A 7-module Streamlit application supporting single-image analysis, live camera capture, batch processing, dataset exploration, benchmark analytics, and downloadable PDF-ready diagnostic certificates.

---

## 2. Dataset Overview & Class Distribution

The model is trained, validated, and evaluated on an authentic, deduplicated benchmark dataset comprising **4,878 wheat images** across 5 distinct categories:

| Class Name | Scientific Classification | Dataset Count | Share (%) | Pathology Category |
| :--- | :--- | :---: | :---: | :--- |
| **Healthy** | *Triticum aestivum* | 1,579 | 32.37% | Non-pathological normal tissue |
| **Yellow Rust** | *Puccinia striiformis f. sp. tritici* | 1,395 | 28.60% | Fungal Basidiomycete (Stripe rust) |
| **Brown Rust** | *Puccinia triticina* | 1,256 | 25.75% | Fungal Basidiomycete (Leaf rust) |
| **Septoria** | *Zymoseptoria tritici* | 349 | 7.15% | Fungal Ascomycete (Leaf blotch) |
| **Loose Smut** | *Ustilago tritici* | 299 | 6.13% | Seed-borne Basidiomycete (Inflorescence smut) |
| **Total** | — | **4,878** | **100.0%** | **5 Target Classes** |

### Addressing Class Imbalance
Classes such as **Loose Smut** (299 images) and **Septoria** (349 images) have significantly fewer samples than **Healthy** (1,579 images). Left unaddressed, standard neural networks bias predictions toward dominant classes. To overcome this:
- **Inverse Frequency Class Weighting**: Computed dynamically using scikit-learn:
  $$w_j = \frac{N}{K \cdot n_j}$$
  where $N$ is total samples (4,878), $K$ is number of classes (5), and $n_j$ is samples in class $j$.
- **On-the-Fly Spatial Data Augmentation**: Horizontal flip, random rotation ($\pm 8\%$), random zoom ($\pm 10\%$), random translation ($\pm 5\%$), and contrast jitter ($\pm 10\%$).

---

## 3. Deep Learning Architecture Specification

The improved architecture is an engineered 4-block deep convolutional network:

```
Input Tensor: (Batch_Size, 224, 224, 3)
   │
   ▼
[Data Augmentation Layer] (Horizontal Flip, Rotation, Zoom, Contrast)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Block 1: Conv2D (32, 3x3) -> BatchNorm -> Conv2D (32) │ -> MaxPool (2x2) -> Dropout(0.15)
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Block 2: Conv2D (64, 3x3) -> BatchNorm -> Conv2D (64) │ -> MaxPool (2x2) -> Dropout(0.20)
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: Conv2D (128, 3x3)-> BatchNorm -> Conv2D (128)│ -> MaxPool (2x2) -> Dropout(0.25)
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│ Block 4: Conv2D (256, 3x3)-> BatchNorm -> Conv2D (256)│ -> MaxPool (2x2) -> Dropout(0.30)
└────────────────────────────────────────────────────────┘
   │
   ▼
[GlobalAveragePooling2D] (Reduces 14x14x256 to 256 vector)
   │
   ▼
[Dense Layer: 128 units, ReLU, L2 Regularization (1e-4)]
   │
   ▼
[Batch Normalization + Dropout(0.35)]
   │
   ▼
[Dense Output Layer: 5 units, Softmax Activation]
```

### Key Hyperparameters
- **Input Dimensions:** $224 \times 224 \times 3$ (RGB)
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam ($\text{learning rate} = 10^{-4}$)
- **Batch Size:** 32
- **Epochs:** 35 with EarlyStopping & ReduceLROnPlateau
- **Kernel Regularizer:** L2 weight decay ($10^{-4}$) on all convolutional & dense kernels

---

## 4. Explainable AI (Grad-CAM) Mathematical Formulation

Deep CNNs are often criticized as "black-box" classifiers. To validate that the network detects actual fungal pustules rather than background soil or lighting artifacts, **Gradient-weighted Class Activation Mapping (Grad-CAM)** was integrated:

1. **Gradient Computation**: Compute the gradient of the unnormalized score for target class $c$ ($y^c$) with respect to feature activation map $A^k$ of the final convolutional layer (`conv2d_7`):
   $$\frac{\partial y^c}{\partial A^k_{i, j}}$$

2. **Global Average Pooling of Gradients**: Obtain neuron importance weights $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i, j}}$$
   where $Z = U \times V$ represents the spatial dimensions of the feature map ($28 \times 28$).

3. **Linear Combination & Rectified Linear Unit (ReLU)**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
   ReLU ensures only features positively contributing to class $c$ are retained, filtering out pixels that decrease class confidence.

4. **Bilinear Upsampling & Superimposition**: The resulting $28 \times 28$ heatmap is resized to $224 \times 224$, colorized with the `Jet` colormap, and alpha-blended ($\alpha = 0.45$) onto the input leaf image.

---

## 5. Experimental Results & Ablation Study

Evaluation on the independent, unseen **488-image stratified test set**:

### Comparative Benchmark
| Metric | Baseline CNN | Improved Deep CNN | Academic Delta |
| :--- | :---: | :---: | :---: |
| **Overall Top-1 Accuracy** | **84.63%** | **89.14%** | **+4.51%** |
| **Macro Average F1-Score** | 0.7964 | 0.8608 | **+0.064** |
| **Weighted Average F1-Score** | 0.8494 | 0.8920 | **+0.043** |
| **Loose Smut Precision** | 58.97% | **88.89%** | **+29.92% (Major Gain)** |
| **Septoria F1-Score** | 68.49% | **73.68%** | **+5.19%** |
| **Yellow Rust Precision** | 99.10% | 90.91% | Balanced recall (+7.14%) |

### Per-Class Performance Matrix (Improved Model)
| Pathology Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Brown rust** | 0.9417 | 0.9040 | 0.9224 | 125 |
| **Healthy** | 0.8929 | 0.9494 | 0.9202 | 158 |
| **Loose Smut** | 0.8889 | 0.8000 | 0.8421 | 30 |
| **Septoria** | 0.6829 | 0.8000 | 0.7368 | 35 |
| **Yellow rust** | 0.9091 | 0.8571 | 0.8824 | 140 |
| **Total / Weighted Avg** | **0.8947** | **0.8914** | **0.8920** | **488** |

---

## 6. Application Architecture & Features

The user-facing application is structured into 7 core modules:
1. **🏠 Overview & Dashboard:** Executive project summary, key performance indicators, system architecture diagram, and diagnostic launcher.
2. **🔬 AI Disease Diagnosis:** Single-specimen diagnostic workflow supporting file upload, live webcam snapshot, or 1-click test sampling from verified dataset images. Generates class probabilities, Grad-CAM visual heatmaps, severity grade (<15% Mild, 15–35% Moderate, >35% Severe), agronomic action plan, and downloadable HTML/PDF report.
3. **📁 Batch Image Diagnosis:** High-throughput batch inference pipeline. Processes multiple images concurrently, calculates epidemiological class distributions, and exports CSV reports.
4. **🗂️ Dataset Explorer & Analytics:** Live scanning and visualization of `D:\non_dup_dataset`, showcasing class distribution, imbalance mitigation rationale, and interactive image browser.
5. **📈 Model Performance & Benchmarks:** Comprehensive academic evaluation suite displaying confusion matrix, classification report, accuracy/loss curves, and ablation tables.
6. **📖 Wheat Pathology Encyclopedia:** Scientific agronomy compendium detailing etiology, symptoms, cultural management, organic bio-pesticides, and chemical active ingredients.
7. **🕒 Scan History & Audit Log:** Audit trail tracking all diagnoses performed during the session with filtering and CSV export.

---

## 7. Viva Voce & Oral Examination Preparation Guide

### Q1: Why did you design a custom CNN instead of fine-tuning a pre-trained model like ResNet50 or VGG16?
> **Answer:** "While transfer learning with architectures like ResNet50 is common, those models possess 25M+ parameters and require heavy computational infrastructure. Our custom 4-block CNN has only **1.2M trainable parameters** (~14 MB file size), making it lightweight, fast, and deployable on edge devices or standard agricultural extension computers without needing specialized GPUs. Furthermore, by designing the architecture from scratch, we maintained full mathematical control over regularizers, batch normalization placement, and layer activation extraction for Grad-CAM."

### Q2: How did you handle the significant class imbalance between Healthy (1,579) and Loose Smut (299)?
> **Answer:** "We utilized a dual strategy:
> 1. **Inverse Frequency Class Weighting** via Scikit-Learn: During categorical cross-entropy optimization, misclassifications on minority classes (Loose Smut and Septoria) incurred proportional penalty multipliers, preventing the gradient updates from being dominated by Healthy samples.
> 2. **On-the-Fly Data Augmentation**: Applying horizontal reflections, rotational perturbations ($\pm 8\%$), and contrast variations synthesized spatial variability on minority classes during training, boosting Loose Smut precision from 58.97% to 88.89%."

### Q3: What is Grad-CAM and why is it necessary in this project?
> **Answer:** "Grad-CAM stands for Gradient-weighted Class Activation Mapping. In agricultural and medical image classification, standard CNNs can achieve high accuracy by latching onto spurious correlations (such as soil color, shadow, or leaf orientation) rather than true disease pustules. Grad-CAM computes the partial derivatives of the winning class score with respect to the feature activation maps of our final convolutional layer (`conv2d_7`). By global-average pooling these gradients, we weight each feature map and pass them through a ReLU function to obtain a 2D spatial heatmap showing exactly where the network looked to make its decision."

### Q4: Why did you use GlobalAveragePooling2D instead of Flattening?
> **Answer:** "A traditional `Flatten` layer flattens $14 \times 14 \times 256 = 50,176$ activations into a 1D vector, which, when connected to a dense layer of 128 units, would create over 6.4 million weights in a single layer—leading to severe overfitting and loss of spatial translation invariance. `GlobalAveragePooling2D` computes the mean of each $14 \times 14$ feature slice, reducing the tensor to 256 scalar values with zero trainable parameters. This drastically regularizes the network and enhances spatial interpretability."

### Q5: What is the clinical difference between Brown Rust and Yellow Rust?
> **Answer:** "Both are fungal rusts, but:
> - **Yellow Rust (*Puccinia striiformis*)**: Produces bright yellow-orange pustules arranged in distinct, parallel, linear stripes along the leaf veins, primarily active in cooler temperatures (10°C–15°C).
> - **Brown Rust (*Puccinia triticina*)**: Produces reddish-brown circular to oval pustules scattered randomly across the leaf blade, thriving in warmer temperatures (15°C–25°C)."

---

## 8. Instructions to Launch the Application

### Method 1: One-Click Desktop Launcher
Double-click the file:
```
D:\CNN Project\run_app.bat
```

### Method 2: Command Line (PowerShell / Terminal)
```powershell
cd "D:\CNN Project"
& ".\venv\Scripts\python.exe" -m streamlit run app.py
```

The application will launch automatically in your web browser at:
`http://localhost:8501`
