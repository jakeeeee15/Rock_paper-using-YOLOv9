# Real-Time Rock-Paper-Scissors AI Game Engine

An interactive, real-time Rock-Paper-Scissors game engine powered by computer vision. The project leverages a custom-trained **YOLOv9 Tiny** object detection model deployed via Python and OpenCV to track hand gestures live, matching the player against an AI opponent through an optimized state machine.

## ✨ Features
*   **Continuous Live Tracking:** Real-time bounding box annotations and confidence scores rendered directly on the webcam feed.
*   **Prediction Memory Buffer:** Features a 1-second history fallback cache to prevent round failures caused by sudden motion blur or frame drops at zero-hour.
*   **Class-Specific Confidence Tuning:** Dynamic thresholding to stabilize difficult multi-feature classifications (like scissors) without increasing background false positives.
*   **Hardware Accelerated:** Fully optimized to run inference locally utilizing NVIDIA CUDA.

---

## 📸 Data Pipeline

### 1. Data Collection
*   **Classes:** Three target gestures: `rock`, `paper`, and `scissor`.
*   **Dataset Integrity:** Images were captured using a localized webcam setup under varying lighting conditions and diverse backgrounds to ensure generalizability.
*   **Annotation:** Bounding boxes were manually drawn and labeled tightly around the geometric perimeter of the hand profiles.

### 2. Data Augmentation
To artificiality boost dataset volume, introduce environmental invariance, and protect the model from overfitting, the following augmentations were applied:
*   **Horizontal Flip:** Applied to make the model ambidextrous, ensuring it tracks both left and right-hand structural variations.
*   **Rotation (Up to ±30°):** Guards the model against orientation bias, enabling detection when gestures are tilted or thrown at creative angles.
*   **Brightness & Contrast Adjustments:** Simulates varied exposure levels to ensure robust performance across changing room lighting environments.

---

## 🛠️ Performance & Training
The model was trained locally on custom weights over **50 epochs**.
*   **Convergence Speed:** Rapid feature extraction stabilized by epoch 30.
*   **Final Accuracy:** Achieved a top-tier score of **95.5% mAP50** (Mean Average Precision), delivering highly accurate classification performance.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install ultralytics opencv-python
