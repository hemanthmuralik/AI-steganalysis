# AI-Powered Image Steganalysis

![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![Deep Learning](https://img.shields.io/badge/TensorFlow-CNN-orange)
![Security](https://img.shields.io/badge/Cybersecurity-Steganalysis-red)

A Deep Learning framework designed to **detect hidden data (steganography)** within digital images. Unlike traditional statistical methods, this project utilizes a custom **Convolutional Neural Network (CNN)** to identify invisible noise patterns introduced by LSB (Least Significant Bit) embedding algorithms.

## 🎥 Visual Demo

### 1. The Challenge (Human vs. AI)
Can you spot the difference? One image contains a hidden payload; the other is clean.
| Cover Image (Clean) | Stego Image (Hidden Data) |
| :---: | :---: |
| ![Clean](assets/clean.jpg) | ![Stego](assets/stego.jpg) |
*(To the human eye, these are identical. The AI detects the Stego image with 92% confidence.)*

### 2. What the AI Sees (Difference Map)
![Difference Map](assets/diff_map.jpg)
*Amplified noise artifacts detected by the model.*

## 🧠 System Architecture

The system consists of two modules:
1.  **Steganography Engine:** Generates a synthetic dataset by embedding text/files into images using LSB substitution.
2.  **Steganalysis Network:** A CNN trained to classify images as "Cover" (Clean) or "Stego" (Infected).



## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, PIL
* **Data Analysis:** Pandas, NumPy, Matplotlib

## 📂 Project Structure
