# VigiLens Codebase Documentation

## 1. Project Overview

VigiLens is a multimodal anomaly detection system for surveillance video that combines visible RGB and infrared/thermal camera inputs. The system uses pretrained feature extractors (ResNet50) and distance-based anomaly scoring, requiring no project-specific training.

**Key Characteristics:**
- True multimodal: Separate visible and IR inputs (not fake grayscale RGB)
- No training required: Uses pretrained ImageNet models
- Real-time capable: Optimized for practical inference
- Configurable: Flexible fusion, detection, and smoothing parameters
- Web interface: Streamlit-based UI for easy use

## 2. Final Architecture

```
Visible Input (Video/Directory)
            ↓
    Frame Source (VideoFrameSource/DirectoryFrameSource)
            ↓
    Preprocessing (FramePreprocessor)
            ↓
    Visible Feature Extractor (ResNet50 Pretrained)
            ↓
    Visible Features (2048-dim)
            ↓
    ┌─────────────────┐
    │                 │
    ↓                 ↓
IR Input          Multimodal Fusion
    ↓                 (concat/weighted/average)
Frame Source              ↓
    ↓              Fused Features
Preprocessing              ↓
    ↓         Anomaly Detection
IR Feature Extractor  (Distance-based Scoring)
    ↓                 ↓
IR Features (2048-dim)  Anomaly Score
            ↓                 ↓
            └────────→ Temporal Smoothing
                              ↓
                        Anomaly Decision
                              ↓
                        Streamlit UI
```

## 3. Directory Structure

```
VigiLens/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── Dockerfile                     # Docker configuration
│
├── config/
│   └── config.py                  # System configuration
│
├── models/
│   ├── feature_extractor.py       # Pretrained ResNet50 extractors
│   ├── fusion.py                  # Multimodal fusion strategies
│   └── anomaly_detector.py        # Distance-based anomaly detection
│
├── pipeline/
│   ├── multimodal_pipeline.py     # Main pipeline orchestration
│   ├── frame_source.py            # Frame reading (visible + IR)
│   └── temporal.py                # Temporal smoothing
│
├── utils/
│   ├── preprocessing.py           # Frame preprocessing
│   └── segmentation.py            # YOLO object detection (auxiliary)
│
├── tests/
│   ├── test_preprocessing.py      # Preprocessing tests
│   └── test_fusion.py             # Fusion tests
│
└── docs/
    └── CODEBASE.md                # This documentation
```

### File Descriptions

#### app.py
**Purpose:** Streamlit web application interface.

**Key Components:**
- Sidebar configuration (threshold, fusion method, smoothing)
- Dual file upload (visible + IR videos)
- Real-time video processing display
- Metrics dashboard (score, status, FPS, objects)

**Inputs:** Visible video file, optional IR video file

**Outputs:** Web UI with video display, metrics, and anomaly alerts

**Dependencies:** streamlit, cv2, numpy, config, pipeline, ultralytics

---

#### config/config.py
**Purpose:** Centralized system configuration.

**Key Classes:**
- `ModelConfig`: Feature extractor settings (ResNet50, device)
- `PreprocessingConfig`: Frame size, normalization, IR handling
- `FusionConfig`: Fusion method and weights
- `AnomalyConfig`: Threshold, distance metric, adaptive settings
- `TemporalConfig`: Smoothing method and window size
- `UIConfig`: Display settings
- `SystemConfig`: Main configuration combining all sub-configs

**Usage:** Import and modify config values, or use `get_config()` for defaults

---

#### models/feature_extractor.py
**Purpose:** Pretrained feature extraction using ResNet50.

**Key Classes:**
- `PretrainedFeatureExtractor`: Base ResNet50 extractor (ImageNet pretrained)
- `VisibleFeatureExtractor`: Specialized for visible RGB frames
- `IRFeatureExtractor`: Specialized for IR frames (converts to 3-channel)
- `FeatureExtractorFactory`: Factory for creating extractors

**Model:** ResNet50 pretrained on ImageNet (torchvision)

**Feature Dimension:** 2048

**Device:** CUDA (if available) or CPU

**Limitations:** IR frames are converted to 3-channel to work with RGB encoder. The encoder was not trained on thermal data - this is a pragmatic compromise.

---

#### models/fusion.py
**Purpose:** Multimodal fusion of visible and IR features.

**Key Classes:**
- `MultimodalFusion`: Main fusion class supporting multiple methods
- `LateFusion`: Fusion at feature level (default)

**Fusion Methods:**
1. **concat:** Concatenate feature vectors (2048 + 2048 = 4096)
2. **weighted:** Weighted sum (alpha * visible + beta * IR)
3. **average:** Simple average ((visible + IR) / 2)

**Mathematical Formulation:**
```
Concat: F_fused = concat(F_visible, F_IR)
Weighted: F_fused = α * F_visible + β * F_IR
Average: F_fused = (F_visible + F_IR) / 2
```

---

#### models/anomaly_detector.py
**Purpose:** Distance-based anomaly detection.

**Key Classes:**
- `DistanceAnomalyDetector`: Computes distance from reference features
- `AnomalyScorer: Normalizes scores to [0, 1] range

**Method:**
1. Maintain running average of features as reference
2. Compute distance (euclidean or cosine) from reference
3. Apply threshold to detect anomalies
4. Optional adaptive thresholding

**Distance Metrics:**
- Euclidean: ||features - reference||
- Cosine: 1 - (features · reference) / (||features|| ||reference||)

**No Training Required:** Uses pretrained features directly

---

#### pipeline/multimodal_pipeline.py
**Purpose:** Main pipeline orchestration.

**Key Classes:**
- `MultimodalAnomalyPipeline`: Complete pipeline integration

**Pipeline Stages:**
1. Frame reading (visible + IR)
2. Preprocessing (resize, normalize)
3. Feature extraction (ResNet50)
4. Multimodal fusion
5. Anomaly detection (distance-based)
6. Temporal smoothing

**Key Methods:**
- `load_source()`: Load video/directory sources
- `process_frame()`: Process single frame pair
- `process_video()`: Process entire video
- `reset()`: Reset pipeline state

---

#### pipeline/frame_source.py
**Purpose:** Frame reading from various sources.

**Key Classes:**
- `MultimodalFrameSource`: Abstract base class
- `VideoFrameSource`: Synchronized video files (visible.mp4 + thermal.mp4)
- `DirectoryFrameSource`: Paired frame directories
- `SingleVideoSource`: Visible-only mode

**Input Formats:**
- Separate video files (visible.mp4, thermal.mp4)
- Paired directories (visible/, ir/)
- Single video (visible only)

**Frame Synchronization:** Reads frames in lockstep from both sources

**IR Handling:** Clearly reports IR unavailable if not provided

---

#### pipeline/temporal.py
**Purpose:** Temporal smoothing of anomaly scores.

**Key Classes:**
- `TemporalSmoother`: Main smoothing class

**Smoothing Methods:**
1. **moving_average:** Simple moving average over window
2. **exponential:** Exponential moving average (EMA)
3. **consecutive:** Requires N consecutive anomalous frames

**Purpose:** Reduce noise and prevent false positives from single-frame anomalies

---

#### utils/preprocessing.py
**Purpose:** Frame preprocessing for visible and IR.

**Key Classes:**
- `FramePreprocessor`: Handles preprocessing for both modalities

**Operations:**
- Resize to target size (default 224x224)
- Normalize to [0, 1]
- Standardize using ImageNet statistics (visible)
- Convert IR to 3-channel if needed

**Important:** Does NOT fake IR by converting RGB to grayscale. Expects real thermal data.

**IR Validation:** Includes heuristic checks for thermal data

---

#### utils/segmentation.py
**Purpose:** YOLO object detection (auxiliary).

**Purpose:** Provides object detection and localization as auxiliary information.

**Model:** YOLOv8n (ultralytics)

**Role:** Separate from anomaly detection - provides contextual information

---

#### tests/test_preprocessing.py
**Purpose:** Test preprocessing functionality.

**Tests:**
- Visible frame preprocessing
- IR frame preprocessing
- Pair preprocessing
- IR unavailable handling

**Run:** `python tests/test_preprocessing.py`

---

#### tests/test_fusion.py
**Purpose:** Test fusion functionality.

**Tests:**
- Concatenation fusion
- Weighted fusion
- Average fusion
- IR unavailable handling

**Run:** `python tests/test_fusion.py`

---

## 4. End-to-End Data Flow

```
1. Input
   - Visible video file or directory
   - IR video file or directory (optional)

2. Frame Synchronization
   - FrameSource reads synchronized frame pairs
   - Handles different frame rates/resolutions
   - Reports IR unavailable if missing

3. Preprocessing
   - Resize to 224x224
   - Normalize to [0, 1]
   - Apply ImageNet standardization (visible)
   - Convert IR to 3-channel if needed

4. Feature Extraction
   - Visible: ResNet50 → 2048-dim features
   - IR: ResNet50 → 2048-dim features

5. Fusion
   - Concatenate/weighted/average fusion
   - Output: 4096-dim (concat) or 2048-dim (weighted/average)

6. Anomaly Detection
   - Compute distance from reference features
   - Apply threshold
   - Output: anomaly score + binary decision

7. Temporal Smoothing
   - Moving average/exponential/consecutive
   - Output: smoothed score + smoothed decision

8. Visualization
   - Display visible frame with anomaly indicator
   - Display IR frame (if available)
   - Show metrics (score, status, FPS, objects)
```

## 5. Visible Processing

**Input:** BGR frame from OpenCV (H, W, 3)

**Steps:**
1. Convert BGR to RGB
2. Resize to 224x224
3. Normalize to [0, 1]
4. Standardize using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

**Output:** Preprocessed frame (224, 224, 3)

**Feature Extraction:** ResNet50 pretrained on ImageNet → 2048-dim features

## 6. IR Processing

**Input:** Real IR/thermal frame (H, W) or (H, W, 1)

**Steps:**
1. Validate as single-channel thermal data
2. Resize to 224x224
3. Normalize to [0, 1]
4. Standardize (mean=0.5, std=0.5)
5. Add channel dimension: (H, W, 1)
6. Convert to 3-channel by replication: (H, W, 3)

**Important Limitation:** The IR frames are converted to 3-channel to work with the RGB-pretrained ResNet50 encoder. The encoder was NOT trained on thermal data. This is a pragmatic compromise due to lack of widely-available IR-specific pretrained models.

**Feature Extraction:** Same ResNet50 encoder → 2048-dim features

**Note:** Despite the limitation, the system still benefits from multimodal fusion because IR provides different information (thermal vs RGB) and features are extracted from different modalities.

## 7. Fusion

**Mathematical Formulation:**

Given:
- F_visible: Visible feature vector (2048-dim)
- F_IR: IR feature vector (2048-dim)

**Concatenation:**
```
F_fused = concat(F_visible, F_IR)
Dimension: 2048 + 2048 = 4096
```

**Weighted:**
```
F_fused = α * F_visible + β * F_IR
where α + β = 1
Dimension: 2048
```

**Average:**
```
F_fused = (F_visible + F_IR) / 2
Dimension: 2048
```

**Default:** Concatenation (preserves all information)

## 8. Anomaly Detection

**Model/Method:** Distance-based scoring using pretrained features

**Pretrained Weights:** ResNet50 ImageNet (downloaded automatically by torchvision)

**Feature Representation:** 2048-dim feature vectors from ResNet50

**Anomaly Score:**
- Raw distance from reference features
- Normalized to [0, 1] for interpretation
- Higher score = more anomalous

**Threshold:** Configurable (default: 0.65)

**Temporal Smoothing:**
- Moving average over N frames (default: 10)
- Requires consecutive anomalous frames (default: 3)
- Reduces noise and false positives

**No Training Required:** Uses pretrained features directly with distance-based scoring

## 9. Why ConvLSTM Was Removed

The previous implementation used a ConvLSTM model that required project-specific training:

**Problems with Previous Approach:**
1. Required training on project-specific data (training.npy)
2. Training took significant time and resources
3. Model needed to be trained for each new dataset
4. Training pipeline was complex (vid2array.py, train.py)
5. No guarantee of good performance without extensive tuning

**New Approach Benefits:**
1. No training required - uses pretrained ImageNet models
2. Works immediately without dataset preparation
3. Academically defensible (pretrained features + distance scoring)
4. Simpler and more maintainable
5. Better for research/educational use

**Trade-off:** The new approach may not achieve the same theoretical performance as a well-trained ConvLSTM on a specific dataset, but it is practical, explainable, and doesn't require training resources.

## 10. YOLO

**Role:** Auxiliary object detection and localization

**Model:** YOLOv8n (ultralytics)

**Purpose:**
- Detect objects (people, vehicles, etc.)
- Provide bounding boxes for visualization
- Offer contextual information

**Separation from Anomaly Detection:**
- YOLO object count is NOT used for anomaly scoring
- Anomaly detection uses feature-based distance scoring
- YOLO is purely for visualization and auxiliary context

**Display:** Optional bounding boxes on visible frame

## 11. Streamlit UI

**Major Components:**

**Sidebar Configuration:**
- Anomaly Detection: threshold, distance metric, adaptive threshold
- Temporal Smoothing: method, window size, consecutive frames
- Multimodal Fusion: method, visible weight
- Display: show IR frame, show bounding boxes

**Input Section:**
- Visible video upload (required)
- IR/thermal video upload (optional)

**Main Display:**
- Visible frame with anomaly indicator (red border if anomalous)
- IR frame (if available and enabled)
- Metrics: anomaly score, status, FPS, frame number, IR availability, object count

**Status Box:**
- Green: NORMAL
- Red: ANOMALY DETECTED

**Alert Display:**
- Red border around frame when anomalous
- "ANOMALY" text overlay
- Anomaly score display

## 12. Configuration

**Configurable Parameters:**

**Model:**
- Device (cuda/cpu)
- Pretrained weights

**Preprocessing:**
- Target size (default: 224x224)
- Normalization (on/off)
- IR as RGB (on/off)

**Fusion:**
- Method (concat/weighted/average)
- Visible weight (0.0-1.0)
- IR weight (0.0-1.0)

**Anomaly Detection:**
- Threshold (0.0-1.0)
- Distance metric (euclidean/cosine)
- Adaptive threshold (on/off)
- Threshold window size

**Temporal Smoothing:**
- Method (moving_average/exponential/consecutive)
- Window size (1-30)
- Consecutive frames (1-10)
- Alpha (for exponential smoothing)

**UI:**
- Show IR frame
- Show bounding boxes
- Display FPS, frame number, object count

## 13. Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
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

## 14. Running

```bash
# Run Streamlit application
streamlit run app.py
```

**Expected Inputs:**
- Visible video file (mp4, avi, mov) - REQUIRED
- IR/thermal video file (mp4, avi, mov) - OPTIONAL

**If IR is not provided:**
- System runs in visible-only mode
- Warning displayed in UI
- Anomaly detection uses only visible features

**If IR is provided:**
- System runs in multimodal mode
- Both visible and IR features extracted
- Features fused for anomaly detection

## 15. Testing

```bash
# Run preprocessing tests
python tests/test_preprocessing.py

# Run fusion tests
python tests/test_fusion.py
```

**Test Coverage:**
- Preprocessing: visible, IR, pair handling, IR unavailable
- Fusion: concat, weighted, average, IR unavailable

**Note:** Tests use synthetic/dummy frames. They verify functionality but do not prove anomaly detection accuracy on real data.

## 16. Hardware

**Expected Requirements:**

**CPU:**
- Modern multi-core processor
- Intel i5/i7 or AMD Ryzen 5/7 recommended

**RAM:**
- 8 GB minimum
- 16 GB recommended

**GPU:**
- Optional but recommended for better performance
- NVIDIA GPU with CUDA support
- 4 GB VRAM minimum

**Inference Performance:**
- CPU: ~5-10 FPS (depending on hardware)
- GPU: ~15-30 FPS (depending on GPU)

**Note:** Performance numbers are estimates. Actual performance depends on hardware, video resolution, and configuration.

## 17. Limitations

**Scientific/Engineering Limitations:**

1. **IR Encoder Limitation:**
   - IR frames are processed by RGB-pretrained ResNet50
   - Encoder was not trained on thermal data
   - This is a pragmatic compromise due to lack of IR-specific pretrained models
   - System still benefits from multimodal fusion (different modalities)

2. **No Project-Specific Training:**
   - Uses generic pretrained features
   - Not optimized for specific surveillance scenarios
   - May not achieve optimal performance on specialized datasets

3. **Visible/IR Synchronization:**
   - Assumes frame-level synchronization
   - No geometric calibration between cameras
   - Different resolutions/FOVs not corrected
   - Simple resize-based alignment only

4. **Threshold Sensitivity:**
   - Anomaly threshold is heuristic
   - May require tuning for different scenarios
   - Adaptive thresholding helps but not perfect

5. **False Positives/Negatives:**
   - Distance-based scoring may produce false alarms
   - Temporal smoothing reduces but doesn't eliminate
   - Performance depends on scene complexity

6. **Lighting/Thermal Conditions:**
   - Performance varies with lighting conditions
   - Thermal conditions affect IR quality
   - Extreme conditions may degrade performance

7. **Domain Mismatch:**
   - ImageNet-trained features for surveillance
   - Domain gap may affect feature quality
   - Generic features may not capture surveillance-specific patterns

## 18. Future Improvements

**Potential Enhancements:**

1. **IR-Specific Pretrained Encoder:**
   - Train or obtain IR-specific pretrained model
   - Better thermal feature extraction
   - Improved multimodal fusion

2. **Calibrated Visible/IR Registration:**
   - Geometric calibration between cameras
   - Pixel-perfect alignment
   - Better fusion at spatial level

3. **Multimodal Transformer:**
   - Attention-based fusion
   - Learn cross-modal relationships
   - Better integration of visible and IR

4. **Project-Specific Anomaly Training:**
   - Train anomaly detector on surveillance data
   - Better domain adaptation
   - Improved accuracy

5. **Temporal Transformer/ConvLSTM:**
   - Better temporal modeling
   - Capture long-term dependencies
   - Improved anomaly detection

6. **Better Threshold Calibration:**
   - Statistical threshold selection
   - Adaptive thresholding improvements
   - Scene-specific tuning

7. **Ensemble Methods:**
   - Multiple anomaly detectors
   - Voting/averaging schemes
   - Robustness improvements

8. **Real-Time Camera Integration:**
   - Live camera input support
   - Real-time streaming
   - Deployment optimization

## 19. Summary

VigiLens is a practical, no-training-required multimodal anomaly detection system that:

- Uses pretrained ResNet50 features from ImageNet
- Combines visible and IR/thermal inputs
- Implements distance-based anomaly scoring
- Provides temporal smoothing for robustness
- Offers a user-friendly Streamlit interface
- Is academically defensible for research/educational use

The system is designed to be immediately usable without training while providing a foundation for future enhancements.
