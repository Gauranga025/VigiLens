# VigiLens Phase 1 Architecture

## Overview

Phase 1 of VigiLens implements a hardened multimodal anomaly detection system with proper calibration/inference separation, explicit frame synchronization, and statistically-grounded anomaly scoring. This phase focuses on architectural correctness and modularity without introducing new trainable models.

## Architecture Diagram

```
Visible Video ──┐
               ├── Frame Synchronization ──→ PairedFrame
IR Video ───────┘
                    ↓
            Frame Preprocessing
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Visible Feature          IR Feature
Extraction (ResNet50)    Extraction (ResNet50)
        └───────────┬───────────┘
                    ↓
            Multimodal Fusion (Concatenation)
                    ↓
            Fused Feature Vector
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
 Calibration Phase    Inference Phase
 (Build Reference)    (Detect Anomalies)
        └───────────┬───────────┘
                    ↓
        Distance-based Anomaly Scoring
                    ↓
            Temporal Smoothing
                    ↓
        Final Anomaly Decision

YOLO (Auxiliary Object Detection)
```

## Key Components

### 1. Frame Synchronization (`pipeline/synchronization.py`)

**Purpose:** Explicitly pair visible and IR frames at the same temporal instant.

**Implementation:**
- `FrameSynchronizer`: Handles FPS-based pairing when timestamps unavailable
- `PairedFrame`: Dataclass containing visible frame, IR frame, frame index, and timestamps
- Supports equal FPS streams via direct pairing
- Supports different FPS streams via frame-index/FPS ratio calculation
- Gracefully handles missing IR frames (visible-only mode)

**Assumptions:**
- If exact timestamp metadata is unavailable, assumes videos are temporally aligned at start
- Uses deterministic frame-index/FPS-based pairing as fallback

### 2. Multimodal Fusion (`models/fusion.py`)

**Purpose:** Combine visible and IR features for anomaly detection.

**Implementation:**
- `MultimodalFusion`: Primary fusion class supporting concat, weighted, and average methods
- **Concatenation is the default and recommended method** for Phase 1
- L2-normalization applied before fusion to ensure numerical comparability
- Preserves modularity for future learned fusion models

**Fusion Methods:**
- **Concatenation (default):** `F_fused = concat(F_visible, F_IR)` → dimension 4096
- Weighted: `F_fused = alpha * F_visible + beta * F_IR` → dimension 2048
- Average: `F_fused = (F_visible + F_IR) / 2` → dimension 2048

### 3. Anomaly Detection with Calibration (`models/anomaly_detector.py`)

**Purpose:** Detect anomalies using distance-based scoring with proper calibration/inference separation.

**Critical Change:** Reference is **frozen during inference** to prevent anomalies from contaminating the normal reference.

**Implementation:**
- `DistanceAnomalyDetector`: Distance-based anomaly detector with explicit phases
- `AnomalyScorer`: Statistical calibration using z-score normalization
- Two-phase operation:
  1. **Calibration Phase:** Collects normal samples, builds reference statistics
  2. **Inference Phase:** Detects anomalies against frozen reference

**Calibration Process:**
1. `start_calibration()`: Begin calibration mode
2. `add_calibration_sample(features)`: Add normal frames
3. After `calibration_window_size` samples, `build_reference()` is called automatically
4. Switches to inference mode
5. Reference is now frozen - no updates during inference

**Scoring:**
- Raw distance preserved (euclidean or cosine)
- Normalized score using z-score: `(distance - mean) / std`
- Threshold can be set manually or derived from percentile

### 4. Temporal Decision Logic (`pipeline/temporal.py`)

**Purpose:** Separate frame-level scoring from temporal aggregation and final decision.

**Implementation:**
- `TemporalSmoother`: Three-stage temporal processing
- Separates:
  1. Frame-level anomaly score (input)
  2. Temporal aggregation (smoothing)
  3. Final anomaly decision (thresholding)

**Smoothing Methods:**
- **Moving Average (default):** Average over sliding window
- Exponential Moving Average: EMA with configurable alpha
- Consecutive Frames: Requires N consecutive anomalous frames

**Decision Threshold:**
- Applied to smoothed score for final binary decision
- Configurable via `decision_threshold` parameter

### 5. Pipeline Integration (`pipeline/multimodal_pipeline.py`)

**Purpose:** Orchestrate all components with proper error handling.

**Implementation:**
- `MultimodalAnomalyPipeline`: Main pipeline class
- Auto-calibration on startup (configurable)
- Error handling in `process_frame()` - returns error status on failure
- Calibration distance collection for scorer calibration
- Reset returns to calibration mode

**Error Handling:**
- Frame processing errors caught and logged
- Returns error status in result dictionary
- Pipeline continues processing subsequent frames

### 6. Configuration (`config/config.py`)

**New Parameters:**
- `AnomalyConfig.calibration_window_size`: Number of frames for calibration (default: 100)
- `AnomalyConfig.auto_calibrate`: Auto-start calibration (default: True)
- `FusionConfig.normalize_features`: L2-normalize before fusion (default: True)
- `TemporalConfig.decision_threshold`: Threshold for final decision (default: 0.5)

### 7. Streamlit UI (`app.py`)

**Changes:**
- Removed experimental fusion method selection (fixed to concatenation)
- Added calibration controls:
  - Calibration frames slider
  - Auto-calibrate checkbox
  - Reset calibration button
- Added device selection (cuda/cpu)
- Added calibration mode indicator in metrics
- Calibration status displayed during calibration phase

## Data Flow

### Calibration Phase

1. User uploads visible and IR videos
2. Pipeline auto-calibrates (if enabled)
3. First N frames (default: 100) are processed as normal samples
4. Features extracted and fused
5. Reference built as mean of calibration features
6. Scorer calibrated with calibration distances
7. Switches to inference mode

### Inference Phase

1. Subsequent frames processed
2. Features extracted and fused
3. Distance computed from frozen reference
4. Score normalized using calibration statistics
5. Temporal smoothing applied
6. Final decision based on threshold
7. YOLO runs in parallel for auxiliary object detection

## YOLO Role

YOLO remains **auxiliary** and is **not** the primary anomaly detector.

- Purpose: Object detection and contextual information
- Display: Bounding boxes shown on visible frame
- Object count: Displayed as metric
- **Not used** for anomaly detection decisions

## What Is NOT Implemented in Phase 1

- **No ConvLSTM** - Temporal processing uses simple smoothing
- **No Pix2PixGAN** - No generative model integration
- **No LLVIP pretrained model** - Using ResNet50 ImageNet weights
- **No trainable anomaly model** - Distance-based only
- **No model training pipeline** - No training code
- **No database/vector database** - In-memory only
- **No cloud deployment** - Local only

## Testing

New test files added:
- `tests/test_synchronization.py`: Frame synchronization tests
- `tests/test_anomaly_calibration.py`: Calibration mechanism tests

Existing tests:
- `tests/test_preprocessing.py`: Preprocessing tests
- `tests/test_fusion.py`: Fusion tests

## Limitations

1. **IR Feature Extraction:** Uses RGB-pretrained ResNet50 with IR converted to 3-channel. The encoder was not trained on thermal data, which is a limitation of the no-training approach.

2. **Frame Synchronization:** Assumes temporal alignment at video start when timestamps unavailable. May not handle complex timing mismatches perfectly.

3. **Calibration Data Quality:** Requires sufficient normal frames for calibration. If calibration data contains anomalies, reference will be contaminated.

4. **Distance-Based Detection:** Simple distance metric may not capture complex anomaly patterns. More sophisticated methods (learned models) would be needed for better accuracy.

5. **No Temporal Model:** Simple smoothing instead of learned temporal models (ConvLSTM, Transformer).

## Phase 2 Recommendations

For Phase 2, the following should be implemented:

1. **LLVIP Pretrained Model:** Replace ResNet50 with IR-specific pretrained model for better thermal feature extraction.

2. **Pix2PixGAN:** Introduce generative model for IR-to-visible translation or anomaly reconstruction.

3. **Learned Fusion:** Replace concatenation with trainable multimodal fusion layer.

4. **Temporal Model:** Add ConvLSTM or Transformer for temporal anomaly detection.

5. **Training Pipeline:** Implement training for anomaly detection models.

6. **Dataset Integration:** Add LLVIP dataset downloader and loader.

## Configuration Reference

### Anomaly Detection
- `anomaly_threshold`: 0.65 (distance threshold)
- `distance_metric`: "euclidean" or "cosine"
- `adaptive_threshold`: False (use adaptive thresholding)
- `calibration_window_size`: 100 (frames for calibration)
- `auto_calibrate`: True (auto-start calibration)

### Temporal Smoothing
- `smoothing_method`: "moving_average" (default)
- `window_size`: 10 (smoothing window)
- `consecutive_frames`: 3 (for consecutive method)
- `alpha`: 0.3 (EMA smoothing factor)
- `decision_threshold`: 0.5 (final decision threshold)

### Fusion
- `fusion_method`: "concat" (fixed in Phase 1)
- `normalize_features`: True (L2-normalize before fusion)

## Performance Considerations

- Feature extraction uses pretrained ResNet50 - efficient inference
- No model training required - fast startup
- Streaming-based processing - does not store entire videos in memory
- Models instantiated once per pipeline - not per frame
- Inference mode (no-grad) used for PyTorch models

## Safety Features

- Reference frozen during inference - prevents anomaly contamination
- Explicit calibration phase - clear separation of normal/anomaly
- Error handling in pipeline - graceful degradation on failures
- Reset functionality - can recalibrate for new environments
- Visible-only fallback - works without IR input
