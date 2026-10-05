# VigiLens: Multimodal Anomaly Detection with Causal Streaming LSTM

---

## 📌 Overview

VigiLens is a deep learning-based video anomaly detection system that learns **normal pedestrian behavior** and detects anomalies as deviations from learned temporal dynamics.

### 🔗 Core Components:

* 🎯 YOLO segmentation for pedestrian ROI extraction
* 🧠 ResNet50 feature extraction (RGB + IR)
* 🌗 Modality-aware fusion (supports RGB+IR, RGB-only, IR-only)
* ⏱️ Causal streaming LSTM for temporal modeling
* 🔮 Next-feature prediction for anomaly detection
* 📊 Statistical calibration for anomaly thresholding

---

## 🎯 Architecture

### Pipeline

```text
RGB frame
    ↓
YOLO Segmentation
    ↓
RGB ROI / mask
    ↓
RGB ResNet50 feature extractor
    ↓
2048-D RGB feature
    |
    |                         IR frame
    |                            ↓
    |                  RGB-derived aligned ROI/mask
    |                            ↓
    |                  IR preprocessing
    |                            ↓
    |                  IR ResNet50 feature extractor
    |                            ↓
    |                         2048-D
    |                            |
    +-------------+--------------+
                  |
                  v
        Modality-aware fusion
                  |
                  v
              512-D feature
                  |
                  v
       Causal streaming LSTM
                  |
                  v
        Normal next-feature
           prediction head
                  |
                  v
        Predicted next feature
                  |
                  v
Compare predicted feature with
actual next feature
                  |
                  v
          Prediction error
                  |
                  v
       Calibrated anomaly score
                  |
                  v
        Normal / Anomaly
```

---

## ⚙️ Methodology

### 1️⃣ YOLO Segmentation

* Uses **YOLOv8n-seg** for instance segmentation
* Focuses on **person/pedestrian class**
* Policy: Combine all detected pedestrian masks into one ROI
* Provides segmentation mask for feature extraction ROI

### 2️⃣ Feature Extraction

* **RGB**: ResNet50 (ImageNet pretrained) → 2048-D features
* **IR**: ResNet50 (ImageNet pretrained) → 2048-D features
  * IR frames converted to 3-channel for RGB encoder
  * **Limitation**: Encoder not trained on thermal data
* Both extractors support ROI-based extraction using segmentation masks

### 3️⃣ Modality-Aware Fusion

* Learnable projections for RGB and IR features
* Supports three modes:
  * **RGB + IR**: Both modalities contribute
  * **RGB only**: RGB embedding dominates
  * **IR only**: IR embedding dominates
* Fixed output dimension: 512-D (regardless of missing modalities)
* Uses zero embeddings for unavailable modalities

### 4️⃣ Causal Streaming LSTM

* **2-layer unidirectional LSTM**
* Input size: 512, Hidden size: 256, Dropout: 0.2
* **Stateful streaming inference**: Hidden state persists across frames
* **No future frames**: Causal behavior for real-time processing
* Hidden state reset when new stream/video/session begins

### 5️⃣ Next-Feature Prediction

* LSTM hidden state → prediction head → predicted next 512-D feature
* **Training objective**: Predict next feature from current feature
* **Loss**: Smooth L1 (robust to outliers)
* **Normal-only training**: Model learns normal temporal dynamics

### 6️⃣ Anomaly Detection

* **Anomaly score = prediction error** (distance between predicted and actual next feature)
* Normal behavior: Low prediction error
* Abnormal behavior: High prediction error (deviates from learned normal dynamics)
* **Calibration**: Threshold set from normal validation data (mean + k*std)

---

## 📁 Project Structure

```bash
VigiLens/
│
├── models/
│   ├── segmentation.py       # YOLO segmentation
│   ├── feature_extractor.py  # ResNet50 RGB/IR extractors
│   ├── fusion.py              # Modality-aware fusion
│   ├── temporal_lstm.py       # Causal streaming LSTM
│   └── anomaly_model.py       # Top-level VigiLens model
│
├── training/
│   ├── dataset.py             # Normal-only video dataset
│   ├── train.py               # Training loop
│   ├── validate.py            # Validation
│   └── calibration.py         # Anomaly calibration
│
├── pipeline/
│   ├── frame_source.py        # Video frame reading
│   ├── synchronization.py     # Frame synchronization
│   ├── preprocessing.py       # Frame preprocessing
│   └── inference.py           # Stateful inference pipeline
│
├── config/
│   └── config.py              # System configuration
│
├── tests/
│   ├── test_fusion.py         # Fusion tests
│   ├── test_lstm.py           # LSTM tests
│   ├── test_inference.py      # Inference tests
│   ├── test_calibration.py    # Calibration tests
│   ├── test_synchronization.py # Synchronization tests
│   └── test_preprocessing.py  # Preprocessing tests
│
├── checkpoints/
│   ├── best_model.pth         # Trained model checkpoint
│   └── calibration.json       # Calibration statistics
│
├── app.py                     # Streamlit UI
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏋️ Training

### Dataset Structure

Organize your training and validation videos as follows:

```bash
Data/
├── train/
│   ├── rgb/
│   │   ├── scene001_rgb.mp4
│   │   ├── scene002_rgb.mp4
│   │   └── ...
│   └── ir/
│       ├── scene001_ir.mp4
│       ├── scene002_ir.mp4
│       └── ...
└── val/
    ├── rgb/
    │   ├── scene010_rgb.mp4
    │   └── ...
    └── ir/
        ├── scene010_ir.mp4
        └── ...
```

**Important notes:**
- RGB and IR videos are paired deterministically by **sorted list order**, not by filename
- Use consistent naming (e.g., `scene001_rgb.mp4` pairs with `scene001_ir.mp4`) to avoid confusion
- `train/` and `val/` directories represent the manual train/validation split
- Training videos must contain **only normal pedestrian behavior** (walking, standing)
- RGB and IR videos must be synchronized (same FPS and frame count)
- Evaluation anomaly videos belong in `Evaluation/` and are not used during training/calibration

### Training Data

* **Normal pedestrian behavior only**: walking, standing
* No anomaly labels required
* Split by VIDEO (not by frames) to avoid data leakage

### Training Command

```bash
python training/train.py
```

The training script automatically discovers videos in `Data/train/` and `Data/val/`.

### Training Configuration

* Sequence length: 16 frames
* Batch size: 4 (configurable)
* Learning rate: 1e-4
* Optimizer: Adam
* Loss: Smooth L1
* Modality dropout: 15% RGB, 15% IR (for robustness)

---

## � Calibration

After training, calibrate anomaly threshold on held-out normal data:

```bash
python training/calibration.py
```

Calibration computes:
* Mean normal error
* Standard deviation
* Threshold: mean + k*std (default k=3)
* Saves to `checkpoints/calibration.json`

---

## ▶️ Inference

### Streamlit UI

```bash
streamlit run app.py
```

The UI supports:
* RGB + IR video upload
* RGB-only mode
* IR-only mode
* Checkpoint and calibration loading
* Real-time anomaly detection visualization

### Inference Behavior

* **Causal**: No future frames used
* **One-frame latency**: Prediction error calculated when next frame arrives
* **Stateful**: LSTM hidden state persists across frames
* **Temporal smoothing**: Moving average or exponential smoothing
* **Threshold decision**: Final anomaly decision based on calibrated threshold

---

## � Application Mode

**Current implementation**: Uploaded video processing via Streamlit UI

The architecture is designed for streaming but currently processes uploaded videos. Live camera support can be added by implementing a live `FrameSource`.

---

## 🚧 Limitations

1. **IR Feature Extraction**: Uses RGB-pretrained ResNet50 (not thermal-specific)
2. **Frame Synchronization**: Assumes temporal alignment at video start when timestamps unavailable
3. **Calibration Quality**: Requires sufficient normal frames for calibration
4. **Distance-Based Detection**: Simple prediction error may not capture complex patterns
5. **IR Mask Alignment**: RGB-derived segmentation mask is resized to thermal frame resolution; true geometric camera calibration is not currently implemented

---

## � Future Work (Phase 2)

* LLVIP pretrained model for IR-specific feature extraction
* Pix2PixGAN for IR-to-visible translation
* Trainable multimodal fusion layer
* ConvLSTM or Transformer for temporal modeling
* Training pipeline implementation
* LLVIP dataset integration
* Live camera support

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
* torch >= 2.0.0
* torchvision >= 0.15.0
* ultralytics >= 8.0.0 (YOLO)
* opencv-python >= 4.8.0
* streamlit >= 1.28.0
* tqdm >= 4.65.0

---

## 📬 Contact

For collaboration or queries, feel free to reach out.
