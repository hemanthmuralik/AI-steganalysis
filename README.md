# Edge-AI Steganalysis (Ye-Net Enhanced)

This repository implements an edge‑optimized steganalysis system using **Ye-Net**, **High-Pass Filters**, **TLU activations**, and **Detection Error (P_E)** as the main evaluation metric.

---

## 🚀 Run Ye-Net Steganalysis on Google Colab

Train, validate, and export the model directly in your browser — no setup needed.

### ▶️ Open the Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthmuralik/Edge-AI-Steganalysis-Security/blob/main/colab/YeNet_Colab.ipynb)

---

## 📘 What the Colab Notebook Includes

### 1. Setup  
- Installs TensorFlow, ONNX, and required libs  
- Clones this repo automatically  

### 2. Load Ye-Net  
```python
from models.ye_net import build_yenet
model = build_yenet()
model.summary()
```

### 3. Preprocessing  
✔ High-Pass Filters (HPF)  
✔ Truncated Linear Unit (TLU)  

### 4. Training With Detection Error (P_E)  
Accuracy is misleading in steganalysis.  
Instead, Colab computes:

- **False Positive Rate (FPR)**
- **False Negative Rate (FNR)**
- **Detection Error:**  
  \[
  P_E = 0.5 	imes (FPR + FNR)
  \]

The model with **lowest P_E** is saved as `best_pe.h5`.

### 5. Inference Demo  
Upload your own images (cover/stego) and see predictions + residual maps.

### 6. Export to Edge Formats  
One-click export to:

- **TFLite** (Android / Raspberry Pi)
- **ONNX** (Jetson / Desktop / Web)

---

## 📂 Notebook Path in Repo

```
colab/YeNet_Colab.ipynb
```

---

