# VigiLens Setup Guide

This guide explains how to set up the local environment for VigiLens after cloning the repository.

## Overview

The VigiLens repository contains only source code, configuration, tests, and documentation. Large files such as datasets, videos, model weights, and virtual environments are intentionally excluded from Git to keep the repository size manageable.

## Files Intentionally Excluded from Git

The following are ignored by `.gitignore` and must be set up locally:

- **Virtual environments:** `env/`, `venv/`, `.venv/`
- **Datasets and videos:** `Data/`, `*.mp4`, `*.avi`, `*.mov`
- **Model weights:** `yolov8n.pt`, `*.pt`, `*.pth`, `*.onnx`, `*.h5`
- **Python cache:** `__pycache__/`, `*.pyc`
- **Generated outputs:** `results/`, `outputs/`, `*.gif`
- **Training data:** `*.npy`, `training.npy`
- **IDE/editor:** `.idea/`, `.vscode/` (except config files)
- **OS files:** `.DS_Store`, `Thumbs.db`
- **Logs:** `*.log`, `logs/`
- **Environment secrets:** `.env`, `.env.*`

## Local Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy>=1.23.5
- opencv-python>=4.8.0
- torch>=2.0.0
- torchvision>=0.15.0
- streamlit>=1.28.0
- ultralytics>=8.0.0
- pillow>=10.0.0

### 3. Download Pretrained Models

**YOLO Model (for object detection):**
```bash
# The YOLOv8n model is automatically downloaded by ultralytics on first use
# Alternatively, download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**ResNet50 Model (for feature extraction):**
- The ResNet50 ImageNet pretrained weights are automatically downloaded by torchvision on first use
- No manual download required
- Downloaded to: `~/.cache/torch/hub/checkpoints/` (Linux/Mac) or `%USERPROFILE%\.cache\torch\hub\checkpoints\` (Windows)

### 4. Place Video Data

**Directory Structure:**
```
VigiLens/
├── Data/
│   ├── Train/          # Training videos (optional)
│   └── Test/           # Test videos (optional)
├── visible_video.mp4   # Visible video file (for testing)
└── thermal_video.mp4   # IR/thermal video file (for testing)
```

**Where to place videos:**
- For testing with the Streamlit UI: Place visible and IR video files in the repository root or any accessible directory
- For batch processing: Organize videos in `Data/Train/` or `Data/Test/` directories
- The system accepts both video files and directories of frames

**Supported Video Formats:**
- MP4 (`*.mp4`)
- AVI (`*.avi`)
- MOV (`*.mov`)
- MKV (`*.mkv`)
- WebM (`*.webm`)

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

### 6. Upload Videos in the UI

1. Click "Browse files" under "Visible Video" to upload a visible RGB video
2. Optionally, click "Browse files" under "IR/Thermal Video" to upload an IR/thermal video
3. Adjust configuration settings in the sidebar as needed
4. The system will process the video and display results

## Reproducing the Local Environment

If you need to reproduce the exact local environment:

```bash
# Export current environment (optional)
pip freeze > requirements-lock.txt

# On a new machine:
python -m venv env
env\Scripts\activate  # Windows
pip install -r requirements-lock.txt
```

## Data Directory Structure

The `Data/` directory is ignored by Git but should be organized as follows:

```
Data/
├── Train/
│   ├── visible/
│   │   ├── scene1.mp4
│   │   └── scene2.mp4
│   └── ir/
│       ├── scene1.mp4
│       └── scene2.mp4
└── Test/
    ├── visible/
    │   └── test1.mp4
    └── ir/
        └── test1.mp4
```

**Note:** The system can also work with paired frame directories:

```
Data/
└── paired_frames/
    ├── visible/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   └── ...
    └── ir/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ...
```

## Model Weights Location

**YOLO Model:**
- Default location: Repository root (`yolov8n.pt`)
- Or ultralytics cache: `~/.config/Ultralytics/` (Linux/Mac) or `%USERPROFILE%\.config\Ultralytics\` (Windows)

**ResNet50 Model:**
- Automatically downloaded by torchvision
- Location: `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth` (Linux/Mac) or `%USERPROFILE%\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth` (Windows)

## Troubleshooting

**Issue:** "Module not found" errors
- **Solution:** Ensure virtual environment is activated and dependencies are installed

**Issue:** CUDA out of memory
- **Solution:** Edit `config/config.py` and set `device = "cpu"` or use a smaller batch size

**Issue:** Model download fails
- **Solution:** Check internet connection, or manually download models and place in the appropriate cache directory

**Issue:** Video file not found
- **Solution:** Ensure video files are in the correct location and paths are correct

**Issue:** IR video not recognized
- **Solution:** Ensure IR video is a valid video file. The system will warn if IR is unavailable and run in visible-only mode

## Additional Notes

- The `Data/` directory is intentionally excluded from Git to avoid committing large video files
- Model weights are excluded from Git because they are large and can be downloaded automatically
- Virtual environments are excluded because they are machine-specific
- Generated outputs (results, extracted frames) are excluded to keep the repository clean
- The `.gitignore` file can be customized if you need to track additional files locally

## Git Workflow

After setting up the local environment:

```bash
# Check status
git status

# Add new source code
git add app.py
git commit -m "Update app.py"

# Push to remote
git push
```

Large files (videos, models, virtual environment) will not be tracked by Git due to `.gitignore` rules.
