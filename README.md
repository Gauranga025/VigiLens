# VigiLens: Multimodal Anomaly Detection with a Causal Streaming LSTM

VigiLens is a deep-learning video anomaly detection system for pedestrian scenes. It is trained **only on normal behavior** (walking, standing) and learns how pedestrian appearance evolves over time. At inference, anything the model cannot predict well, meaning a large gap between the predicted and the actual next-frame feature, is flagged as an anomaly.

It works with paired **visible (RGB) + infrared/thermal (IR)** video, and also with RGB-only or IR-only input.

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Calibration](#calibration)
- [Running the Demo App](#running-the-demo-app)
- [Docker](#docker)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Key Features

- **Normal-only training.** No anomaly labels are required.
- **Pedestrian-focused features.** YOLOv8n-seg masks restrict feature extraction to people.
- **Modality-aware fusion.** RGB+IR, RGB-only and IR-only inputs all produce a fixed 512-D embedding. Missing modalities are replaced by zero embeddings.
- **Causal, stateful inference.** A unidirectional LSTM keeps its hidden state across frames and never looks at future frames.
- **Calibrated thresholds.** The anomaly threshold is derived from held-out normal data (`mean + k·std`).
- **Streamlit UI.** Upload videos and watch the anomaly score, status and FPS update frame by frame.

---

## How It Works

```
RGB frame ─► YOLOv8n-seg ─► person ROI/mask ─► ResNet50 (RGB) ─► 2048-D ─┐
                                   │                                      │
IR frame ───────────────► aligned ROI + IR preprocessing                  ├─► Modality-aware
                                   └─► ResNet50 (IR) ───────► 2048-D ─────┘   fusion (512-D)
                                                                                  │
                                                                     Causal streaming LSTM
                                                                                  │
                                                                   Next-feature prediction head
                                                                                  │
                                         |predicted next feature − actual next feature|
                                                                                  │
                                              temporal smoothing ─► calibrated threshold
                                                                                  │
                                                                         Normal / Anomaly
```

| Stage | Details |
|---|---|
| **Segmentation** | YOLOv8n-seg, person class only. All detected pedestrian masks are merged into a single ROI. |
| **Feature extraction** | ImageNet-pretrained ResNet50 for each modality, giving 2048-D features. IR frames are converted to 3 channels. |
| **Fusion** | Learnable projections per modality, fused to 512-D. Supports `RGB+IR`, `RGB only` and `IR only`. |
| **Temporal model** | 2-layer unidirectional LSTM (input 512, hidden 256, dropout 0.2). The hidden state is reset at the start of each new stream. |
| **Prediction** | The LSTM hidden state feeds a head that predicts the next 512-D feature. Trained with Smooth L1 loss. |
| **Scoring** | Anomaly score = prediction error. Calibrated threshold = `mean + k·std` of errors on normal validation data (default `k = 3`). |

**Latency:** because the score compares a prediction with the *next* feature, each frame's error is available one frame later.

---

## Project Structure

```
VigiLens/
├── app.py                   # Streamlit UI
├── Dockerfile
├── requirements.txt
├── config/
│   └── config.py            # System configuration (SystemConfig, get_config)
├── models/
│   ├── segmentation.py      # YOLO person segmentation
│   ├── feature_extractor.py # ResNet50 RGB/IR extractors
│   ├── fusion.py            # Modality-aware fusion
│   ├── temporal_lstm.py     # Causal streaming LSTM
│   └── anomaly_model.py     # Top-level VigiLensModel
├── pipeline/
│   ├── frame_source.py      # Video frame reading
│   ├── synchronization.py   # RGB/IR frame synchronization
│   ├── preprocessing.py     # Frame preprocessing
│   └── inference.py         # Stateful InferencePipeline
├── training/
│   ├── dataset.py           # Normal-only paired-video dataset
│   ├── train.py             # Training loop
│   ├── validate.py          # Validation
│   └── calibration.py       # AnomalyCalibrator
├── tests/                   # Unit tests (fusion, LSTM, inference, calibration, sync, preprocessing)
└── checkpoints/             # Created by you: best_model.pth, calibration.json (not tracked)
```

---

## Getting Started

### Requirements

- Python 3.8+ (PyTorch 2.x does not support older versions)
- A CUDA-capable GPU is recommended; CPU works but is slow

### Installation

```bash
git clone https://github.com/Gauranga025/VigiLens.git
cd VigiLens

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `ultralytics` (YOLO), `opencv-python`, `streamlit`, `numpy`, `pillow`, `tqdm`.

> The repository does not include trained weights. Train a model and calibrate it (below) before running the app, otherwise the app falls back to an untrained model and a default threshold.

---

## Training

### 1. Prepare the dataset

Place paired RGB and IR videos under `Data/`:

```
Data/
├── train/
│   ├── rgb/  scene001_rgb.mp4  scene002_rgb.mp4  ...
│   └── ir/   scene001_ir.mp4   scene002_ir.mp4   ...
└── val/
    ├── rgb/  scene010_rgb.mp4  ...
    └── ir/   scene010_ir.mp4   ...
```

Rules to follow:

- Videos must contain **normal behavior only** (e.g. walking, standing).
- RGB and IR files are paired by **sorted order, not by filename**. Use matching, consistently sorted names.
- Each RGB/IR pair must be synchronized, with the same FPS and frame count.
- Split by **video**, not by frame, to avoid leakage between `train/` and `val/`.
- Keep anomalous evaluation videos out of `train/` and `val/` (for example in a separate `Evaluation/` folder).

### 2. Train

```bash
python training/train.py
```

The script discovers videos in `Data/train/` and `Data/val/` automatically.

| Setting | Value |
|---|---|
| Sequence length | 16 frames |
| Batch size | 4 (configurable) |
| Optimizer / LR | Adam, 1e-4 |
| Loss | Smooth L1 |
| Modality dropout | 15% RGB, 15% IR (for robustness to missing modalities) |

---

## Calibration

After training, compute the anomaly threshold on held-out **normal** data:

```bash
python training/calibration.py
```

This computes the mean and standard deviation of normal prediction errors, sets the threshold to `mean + k·std` (default `k = 3`), and writes the result to `checkpoints/calibration.json`.

---

## Running the Demo App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (by default <http://localhost:8501>).

**Sidebar settings**

- **Device:** `cuda` or `cpu` (the default is `cuda`, so switch to `cpu` on machines without a GPU)
- **Checkpoint / calibration paths:** default to `checkpoints/best_model.pth` and `checkpoints/calibration.json`
- **Temporal smoothing:** `moving_average` or `exponential`, with a window of 1–30 frames
- **Display:** toggle the IR view and the segmentation mask

**Usage**

1. Upload a visible video (`mp4`, `avi` or `mov`). An IR/thermal video is optional.
2. The app processes the video frame by frame. Anomalous frames get a red border and an `ANOMALY` label, and the sidebar panel shows the smoothed score, status, FPS, frame index and IR availability.

If no IR video is uploaded, the app runs in visible-only mode.

> **Current scope:** the app processes *uploaded* videos. The architecture is built for streaming, and live camera support only requires a live `FrameSource` implementation.

---

## Docker

```bash
docker build -t vigilens .
docker run -p 8501:8501 vigilens
```

Streamlit listens on port 8501 inside the container by default. To use the GPU, run with `--gpus all` (requires the NVIDIA Container Toolkit and a CUDA-enabled base image).

---

## Testing

Unit tests live in `tests/` and cover fusion, the LSTM, inference, calibration, synchronization and preprocessing:

```bash
pip install pytest
python -m pytest tests/
```

---

## Limitations

1. **IR features:** the IR encoder is an RGB/ImageNet-pretrained ResNet50, not trained on thermal data.
2. **Synchronization:** when timestamps are unavailable, RGB and IR are assumed to be aligned at the start of the video.
3. **IR mask alignment:** the RGB-derived mask is resized to the IR resolution. True geometric camera calibration is not implemented.
4. **Calibration quality:** thresholds need enough normal frames to be reliable.
5. **Scoring:** plain prediction error may miss subtle or complex anomalies.

---

## Roadmap

- IR-specific feature extraction (e.g. LLVIP-pretrained model)
- IR-to-visible translation (Pix2Pix GAN)
- Trainable multimodal fusion improvements
- ConvLSTM or Transformer temporal models
- LLVIP dataset integration
- Live camera support

---

## License

No license file is currently included in this repository. Add one (e.g. MIT or Apache-2.0) before sharing or accepting contributions.

## Contact

For collaboration or questions, open an issue on [GitHub](https://github.com/Gauranga025/VigiLens/issues).
