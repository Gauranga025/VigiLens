VigiLens: Multi-Modal Video Anomaly Detection using Spatio-Temporal Deep Learning

Overview

VigiLens is a deep learning-based video anomaly detection system designed to identify abnormal events in surveillance footage by modeling both spatial and temporal patterns. Unlike traditional motion-based systems, VigiLens learns normal behavioral dynamics from video sequences and detects anomalies as deviations from learned patterns.

The system integrates:

- Spatio-temporal modeling (ConvLSTM + Conv3D)
- Reconstruction-based anomaly detection
- Object-level segmentation (YOLOv8)
- Multi-modal fusion (Visible + Infrared) [in progress]


Motivation

Traditional surveillance systems rely on handcrafted rules or simple motion detection, which often fail in complex real-world scenarios. This project aims to address these limitations by leveraging deep learning to:

- Learn normal scene behavior automatically
- Detect subtle anomalies beyond simple motion
- Improve robustness under low-light and noisy conditions using multi-modal inputs


Methodology

1. Spatio-Temporal Modeling

The system uses a ConvLSTM-based architecture to capture both:

- Spatial features (objects, scene layout)
- Temporal dynamics (motion patterns, behavior evolution)

The model processes sequences of 10 frames, enabling it to understand temporal continuity.


2. Reconstruction-Based Anomaly Detection

The model is trained only on normal video sequences.

During inference:

- The model reconstructs input sequences
- The reconstruction error (MSE) is computed

Decision Rule:

- Low error → Normal behavior
- High error → Anomalous behavior

This unsupervised approach eliminates the need for labeled anomaly data.


3. Segmentation (Object-Level Focus)

YOLOv8 is used to:

- Detect objects (e.g., people, vehicles)
- Focus the model on regions of interest (ROI)

This reduces noise and improves detection accuracy.


4. Multi-Modal Fusion (Ongoing Work)

To improve robustness, the system is being extended to incorporate:

- Visible spectrum input (RGB)
- Infrared (IR) input

Fusion enables:

- Better performance in low-light conditions
- Improved feature representation under environmental variations

---

System Pipeline

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


Project Structure

VigiLens/
│
├── models/
│   ├── fusion_model.py        # Multi-modal fusion network
│   ├── anomaly_model.py      # Load trained ConvLSTM model
│
├── utils/
│   ├── preprocess.py         # Frame preprocessing functions
│   ├── segmentation.py       # YOLO-based segmentation
│
├── vid2array.py              # Data preprocessing & sequence generation
├── train.py                  # ConvLSTM training pipeline
├── test.py                   # End-to-end anomaly detection
├── requirements.txt
└── README.md


Dataset

The model is trained and evaluated on standard anomaly detection datasets:

- UCSD Pedestrian Dataset
- Avenue Dataset

Dataset Characteristics:

- Training data: Only normal behavior
- Testing data: Normal + anomalous events


Training Details

- Input shape: "(227 × 227 × 10 × 1)"
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 4
- Epochs: 20 (configurable)

Checkpointing is used to save the best performing model:

model/best_model.keras


How to Run

1. Install dependencies

pip install -r requirements.txt

2. Prepare training data

python vid2array.py

3. Train the model

python train.py

4. Run anomaly detection

python test.py


Results & Observations

- The model successfully distinguishes between normal and abnormal sequences
- Reconstruction error provides a reliable anomaly signal
- Temporal modeling significantly improves detection over frame-based methods


Current Status

- ConvLSTM-based anomaly detection implemented
- Training and inference pipeline completed
- Segmentation (YOLOv8) integrated
- Multi-modal fusion (Visible + IR) under development
- Segmentation-aware anomaly localization in progress


Future Work

- Multi-modal training using real IR datasets
- Attention-based fusion mechanisms
- Pixel-level anomaly localization (heatmaps)
- Real-time deployment using webcam streams
- Integration with edge devices for surveillance systems


Key Learnings

- Spatio-temporal modeling using ConvLSTM
- Unsupervised anomaly detection techniques
- Video data preprocessing and sequence modeling
- Real-world challenges in threshold tuning and noise handling
- Multi-modal learning and sensor fusion concepts


Conclusion

VigiLens demonstrates the effectiveness of combining deep learning, temporal modeling, and multi-modal perception for anomaly detection in surveillance systems. The project lays the foundation for robust real-world applications in smart cities, security monitoring, and intelligent surveillance systems.


Contact

For further details or collaboration opportunities, feel free to reach out.