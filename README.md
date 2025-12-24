# DeepFake_Project
# Deepfake Detection using Hybrid Neural Architectures

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-95.62%25-brightgreen)

## 📌 Project Overview
The proliferation of deepfakes (AI-generated synthetic media) poses a significant threat to information integrity. This project implements a **robust, multi-modal deepfake detection system** that differentiates between real and manipulated faces by analyzing **spatial artifacts, frequency anomalies, and global facial inconsistencies**.

Unlike standard CNN-only approaches, this project utilizes a **Weighted Ensemble** of Convolutional Neural Networks (ResNet, Xception), Vision Transformers (ViT), and Frequency-domain classifiers (FFT) to achieve state-of-the-art performance.

**Key Achievement:** The final weighted ensemble achieved **95.62% accuracy** on a balanced dataset of ~13,000 facial images (FaceForensics++ & DFFD).

---

## 🚀 Key Features
* **Hybrid Architecture:** Combines texture-sensitive CNNs (Xception) with global-attention Transformers (ViT).
* **Frequency Domain Analysis:** Implements a **Frequency-CNN** that performs FFT (Fast Fourier Transform) to detect compression artifacts invisible to the human eye.
* **Advanced Optimization:** Utilizes **Mixed Precision (AMP)** for faster training and **Cosine Annealing** for stable convergence.
* **Explainable AI (XAI):** Integrated **Grad-CAM** (for CNNs) and **EigenCAM** (for ViT) to visualize decision boundaries.
* **Weighted Ensemble:** A custom soft-voting mechanism that assigns weights based on validation stability.

---

## 📊 Performance Results

The system was tested on a strictly unseen test set. The **Vision Transformer (ViT)** proved to be the strongest single model, but the **Ensemble** provided the best robustness.

| Model | Accuracy (%) | Key Observation |
| :--- | :--- | :--- |
| **LeNet (Baseline)** | ~61.90% | Underfits; fails to capture fine texture noise. |
| **Frequency-CNN** | 74.30% | Detects compression artifacts but misses spatial cues. |
| **CBAM-ResNet18** | 91.54% | Strong spatial localization due to attention mechanism. |
| **XceptionNet** | 92.29% | Excellent at detecting "blending" boundaries. |
| **Vision Transformer (ViT)** | **93.27%** | Best single model; captures global facial inconsistencies. |
| **Weighted Ensemble** | **95.62%** | **Best Overall**; reduces false positives significantly. |

---

## 🏗️ Architecture & Methodology

### 1. Data Preprocessing
* **Dataset:** 13,000+ images from FaceForensics++ and DFFD (Balanced 50/50 split).
* **Pipeline:** Face Cropping $\rightarrow$ Resize $\rightarrow$ Normalization (Mean/Std: 0.5).
* **Note:** Heavy geometric augmentation (rotation) was avoided as it destroys subtle deepfake texture artifacts.

### 2. Model Pipeline
* **Texture Stream:** **XceptionNet** (Depthwise Separable Convolutions) focuses on high-frequency noise.
* **Attention Stream:** **ViT-B16** (Vision Transformer) analyzes patch-wise consistency across the face.
* **Frequency Stream:** **Freq-CNN** takes the Fourier Transform of the image to spot GAN upsampling artifacts.

### 3. Explainability
We validated that the models focus on the face and not the background:
* **Xception:** Focuses on skin texture and boundaries (high-frequency).
* **ViT:** Distributes attention across the entire face structure.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch, Torchvision
* Scikit-learn, OpenCV
* Timm (PyTorch Image Models)

### Installation
```bash
git clone [https://github.com/yourusername/deepfake-detection.git](https://github.com/yourusername/deepfake-detection.git)
cd deepfake-detection
pip install torch torchvision timm scikit-learn opencv-python matplotlib
