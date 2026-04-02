# Black Pig Project — Depth Estimation Integration
## Context for Claude Code

---

## Project Overview

This is a **Flutter Android application** (`DooMoo/`) that estimates pig weight using **top-down camera images**. The app runs **fully offline** on **low-end Android phones (<5000 THB)**. All ML inference happens on-device.

### Current Pipeline (Already Implemented ✅)

```
Top-down camera image
        ↓
[1] pig_classification/    → Breed classification (TFLite)
        ↓
[2] pig_segmentation/      → Instance segmentation mask (TFLite)
        ↓
[3] transform/             → Perspective correction / homography
        ↓
[4] measure_length/        → Pixel-space length & width from mask
        ↓
[5] transform_model/       → Regression → predicted weight (kg)
        ↓
[DooMoo/]                  → Flutter app — assembles the pipeline
```

### What Is NOT Yet Implemented ❌

**Depth estimation** — currently the user manually measures camera-to-pig distance using an external laser tool and inputs it manually. The goal is to **replace this with an automatic depth model** running on-device.

---

## Your Task

Implement automatic depth/distance estimation so the app can measure **camera height above the pig's back** without any external tool.

The output needed is a **single float value in meters** — e.g. `1.35` — representing the vertical distance from the phone camera to the pig.

---

## Repo Structure

```
URD-2_BlackPigProject/
├── DooMoo/                  # Flutter Android app (main app)
│   ├── lib/                 # Dart source code
│   ├── android/             # Android native (Kotlin/Java + C/C++)
│   └── assets/              # Model files go here (.tflite or .onnx)
├── depth-estimation/        # Python notebooks — model experiments
├── measure_length/          # Length/width calculation notebooks
├── pig_classification/      # Classification training notebooks
├── pig_segmentation/        # Segmentation training notebooks
├── transform/               # Perspective transform notebooks
└── transform_model/         # Weight regression notebooks
```

---

## Technical Constraints

| Constraint | Detail |
|---|---|
| Platform | Android only |
| Connectivity | Fully offline — no network calls |
| Device class | Low-end: Helio G85 / Snapdragon 680 or weaker |
| RAM | ~3–4 GB typical |
| Inference budget | <3 seconds total for full pipeline |
| Camera view | Top-down only (camera pointed straight down at pig) |
| App framework | Flutter (Dart) with native Android (C/C++ via FFI or Kotlin) |
| Existing ML runtime | TFLite (check `DooMoo/` for current usage) |

---

## Approach to Implement

Implement **both** approaches below and let the app pick the best one:

---

### Approach A — Geometric Calibration (Primary, Preferred)

**How it works:**
Since the view is top-down and we already have the pig segmentation mask, we can compute distance purely from geometry:

```
distance_m = (real_pig_length_mm × focal_length_px) / pixel_pig_length
```

- `real_pig_length_mm` → from breed classification output (known average per breed)
- `focal_length_px` → from Android `Camera2` API or calibration
- `pixel_pig_length` → from segmentation mask bounding box (already computed in `measure_length/`)

**Steps to implement:**
1. Read focal length from Android `Camera2` API (`CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS`)
2. Read sensor size (`CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE`)
3. Compute `focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px`
4. Get `pixel_pig_length` from existing segmentation mask (longest axis of bounding box)
5. Map breed → average real body length (lookup table)
6. Compute and return distance in meters

**Files to create/modify:**
- `DooMoo/android/app/src/main/kotlin/.../CameraCalibration.kt` — focal length extraction
- `DooMoo/lib/services/depth_service.dart` — Dart wrapper
- `DooMoo/lib/models/breed_dimensions.dart` — breed → real size lookup table

---

### Approach B — On-Device Depth Model (Fallback)

Use **Depth Anything v2 Small** quantized to **TFLite INT8** as a fallback when geometric calibration confidence is low.

**Model specs:**
- Model: `depth_anything_v2_small` (ViT-S backbone)
- Format: TFLite INT8 quantized
- Input: `256×256` RGB image, normalized `[0, 1]`
- Output: `256×256` float32 depth map (relative, inverse depth)
- Size target: <25 MB

**Steps to implement:**

**Python side (`depth-estimation/`):**
1. Export Depth Anything v2 Small to ONNX
2. Convert ONNX → TFLite using `onnx-tf` + `tf.lite.TFLiteConverter`
3. Apply INT8 post-training quantization with representative dataset
4. Validate output shape and depth range
5. Save to `DooMoo/assets/depth_anything_v2_small_int8.tflite`

**Android/Dart side (`DooMoo/`):**
1. Load TFLite model using `tflite_flutter` package
2. Preprocess input frame: resize to 256×256, normalize
3. Run inference → get `256×256` depth map
4. Apply pig segmentation mask to depth map
5. Compute `median()` of depth values inside mask
6. Convert relative depth → metric using a calibration scale factor
7. Return distance in meters

**Files to create/modify:**
- `depth-estimation/export_depth_model.ipynb` — export + quantize pipeline
- `depth-estimation/validate_depth_model.ipynb` — accuracy vs laser ground truth
- `DooMoo/lib/services/depth_model_service.dart` — TFLite inference wrapper
- `DooMoo/assets/depth_anything_v2_small_int8.tflite` — model file

---

### Approach Selection Logic

```dart
// In DooMoo/lib/services/depth_service.dart

Future<double> estimateDistance({
  required Uint8List imageBytes,
  required SegmentationMask pigMask,
  required String breedLabel,
}) async {
  // Try geometric first (fast, no model needed)
  final geoResult = await _geometricCalibration.estimate(
    pigMask: pigMask,
    breedLabel: breedLabel,
  );

  if (geoResult.confidence > 0.8) {
    return geoResult.distanceMeters;
  }

  // Fall back to depth model
  return await _depthModelService.estimate(
    imageBytes: imageBytes,
    pigMask: pigMask,
  );
}
```

---

## Breed → Real Body Length Lookup Table

Use these as defaults (can be updated from farmer data):

```dart
const Map<String, double> breedRealLengthMm = {
  'black_pig_small':  700.0,   // ~70cm
  'black_pig_medium': 900.0,   // ~90cm
  'black_pig_large':  1100.0,  // ~110cm
  'unknown':          850.0,   // fallback average
};
```

---

## Validation / Testing

Create a validation notebook at `depth-estimation/validate_depth_model.ipynb` that:

1. Loads test images with known laser-measured distances
2. Runs both Approach A and Approach B
3. Computes MAE, RMSE vs laser ground truth
4. Plots error distribution
5. Outputs a comparison table

Target accuracy: **MAE < 15 cm** vs laser measurement.

---

## Key Integration Point

The distance value must plug into the existing weight regression pipeline. Find where `measure_length` output feeds into `transform_model` and add the distance as an additional input feature:

```python
# Current (in regression model)
features = [pig_length_px, pig_width_px]

# New (add distance)
features = [pig_length_px, pig_width_px, distance_m]
```

Check `transform_model/` notebooks to see the current feature vector and update accordingly.

---

## Python Environment (for notebooks)

```bash
pip install torch torchvision
pip install onnx onnxruntime
pip install onnx-tf tensorflow
pip install opencv-python pillow numpy matplotlib
pip install huggingface_hub  # to download Depth Anything v2
```

---

## Flutter Dependencies to Add

In `DooMoo/pubspec.yaml`:

```yaml
dependencies:
  tflite_flutter: ^0.10.4      # TFLite inference
  camera: ^0.10.5              # Camera2 API access
  image: ^4.1.3                # Image preprocessing
```

---

## What to Deliver

1. `depth-estimation/export_depth_model.ipynb` — model export + INT8 quantization
2. `depth-estimation/validate_depth_model.ipynb` — accuracy validation vs laser
3. `DooMoo/android/app/src/main/kotlin/.../CameraCalibration.kt` — focal length from Camera2
4. `DooMoo/lib/services/depth_service.dart` — unified distance service (geo + model)
5. `DooMoo/lib/services/depth_model_service.dart` — TFLite depth model wrapper
6. `DooMoo/lib/models/breed_dimensions.dart` — breed size lookup table
7. Updated `DooMoo/pubspec.yaml` with new dependencies
8. Brief update to `README.md` documenting the depth approach

---

## Notes for Claude Code

- Always check existing files before creating new ones — the segmentation and measure_length modules already extract mask dimensions, reuse them
- The app is Flutter — write Dart code for the service layer, Kotlin only for Camera2 API access
- TFLite is preferred over ONNX Runtime for Android because it's lighter and already likely used in the project
- Do not add internet permissions or any network calls — the app must stay offline
- When in doubt about existing interfaces, read the files in `DooMoo/lib/` first before writing new code
