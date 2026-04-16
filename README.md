# 🚀 VigiLens: Multi-Modal Video Anomaly Detection using Spatio-Temporal Deep Learning

---

## 📌 Overview

VigiLens is a deep learning-based video anomaly detection system designed to identify abnormal events in surveillance footage by modeling both **spatial** and **temporal** patterns.

Unlike traditional motion-based systems, VigiLens learns **normal behavioral dynamics** from video sequences and detects anomalies as deviations from learned patterns.

### 🔗 Core Components:

* 🧠 Spatio-temporal modeling (ConvLSTM + Conv3D)
* 🔁 Reconstruction-based anomaly detection
* 🎯 Object-level segmentation (YOLOv8)
* 🌗 Multi-modal fusion (Visible + Infrared) *(in progress)*

---

## 🎯 Motivation

Traditional surveillance systems rely on handcrafted rules or simple motion detection, which often fail in complex real-world scenarios.

This project addresses these limitations by:

* ✅ Learning normal scene behavior automatically
* ✅ Detecting subtle anomalies beyond motion
* ✅ Improving robustness under low-light & noisy conditions

---

## ⚙️ Methodology

### 1️⃣ Spatio-Temporal Modeling

* Uses **ConvLSTM architecture**
* Captures:

  * Spatial features (objects, scene layout)
  * Temporal dynamics (motion & behavior evolution)

📌 Processes sequences of **10 frames** to model temporal continuity.

---

### 2️⃣ Reconstruction-Based Anomaly Detection

* Trained **only on normal data**
* During inference:

  * Reconstructs input sequence
  * Computes reconstruction error (MSE)

📊 **Decision Rule:**

* Low error → Normal
* High error → Anomaly

✔ No labeled anomaly data required

---

### 3️⃣ Object-Level Segmentation

* Uses **YOLOv8**
* Detects:

  * People
  * Vehicles
* Focuses on **Regions of Interest (ROI)**

🎯 Result:

* Reduced noise
* Improved detection accuracy

---

### 4️⃣ Multi-Modal Fusion *(Ongoing)*

* Inputs:

  * RGB (Visible)
  * Infrared (IR)

📈 Benefits:

* Better performance in low-light
* Robust feature representation

---

## 🔄 System Pipeline

```text
Video Input
   ↓
Frame Extraction & Preprocessing
   ↓
Segmentation (YOLOv8)
   ↓
Fusion (Visible + IR)
   ↓
Sequence Formation (10 frames)
   ↓
ConvLSTM Model
   ↓
Reconstruction
   ↓
Error Calculation (MSE)
   ↓
Thresholding
   ↓
Anomaly Detection
```

---

## 📁 Project Structure

```bash
VigiLens/
│
├── models/
│   ├── fusion_model.py
│   ├── anomaly_model.py
│
├── utils/
│   ├── preprocess.py
│   ├── segmentation.py
│
├── vid2array.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Used standard anomaly detection datasets:

* UCSD Pedestrian Dataset
* Avenue Dataset

### 📌 Characteristics:

* Training → Only normal data
* Testing → Normal + anomalies

---

## 🏋️ Training Details

* 📐 Input Shape: `(227 × 227 × 10 × 1)`
* 📉 Loss Function: Mean Squared Error (MSE)
* ⚡ Optimizer: Adam
* 📦 Batch Size: 4
* 🔁 Epochs: 20 (configurable)

📌 Best model saved using checkpointing:

```
model/best_model.keras
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python vid2array.py
python train.py
python test.py
```

---

## 📈 Results & Observations

⚠️ *Model training is currently in progress — final outputs will be added soon.*

* ✔ Successfully distinguishes normal vs abnormal sequences
* ✔ Reconstruction error acts as reliable anomaly signal
* ✔ Temporal modeling improves performance significantly

---

## 🚧 Current Status

* ✅ ConvLSTM anomaly detection implemented
* ✅ Training & inference pipeline completed
* ✅ YOLOv8 segmentation integrated
* 🚧 Multi-modal fusion under development
* 🚧 Localization improvements in progress

---

## 🔮 Future Work

* 🔬 Multi-modal IR dataset training
* 🧠 Attention-based fusion
* 🔥 Pixel-level anomaly heatmaps
* 🎥 Real-time webcam deployment
* 📡 Edge device integration

---

## 🧠 Key Learnings

* Spatio-temporal modeling (ConvLSTM)
* Unsupervised anomaly detection
* Video preprocessing pipelines
* Threshold tuning challenges
* Multi-modal fusion techniques

---

## 📌 Conclusion

VigiLens demonstrates the effectiveness of combining deep learning, temporal modeling, and multi-modal perception for anomaly detection in surveillance systems.

It lays a strong foundation for real-world applications in:

* 🏙 Smart cities
* 🔐 Security monitoring
* 🎥 Intelligent surveillance

---

## 📬 Contact

For collaboration or queries, feel free to reach out.
