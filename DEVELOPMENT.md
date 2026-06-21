# Developer Guide — Black Pig Project

---

## Pipeline Overview

```
Top-down photo
    ↓
[1] YOLOv8 — Pig detection (bounding boxes) — yolo_detector.dart
    ↓
[2] RF-DETR — Segmentation mask (on cropped region) — pig_detector.dart
    ↓
[3] Perspective correction
    ↓
[4] Pixel-space measurements (length × width)
    ↓
[5] Distance estimation (geometric or depth model)
    ↓
[6] Weight regression → predicted kg
```

All inference runs **on-device** with no network calls.

**Note:** YOLOv8 does initial detection on the full image. RF-DETR runs segmentation on each YOLO-cropped region when the user taps a detected pig.

---

## Repository Structure

```
URD-2_BlackPigProject/
├── DooMoo/                  # Flutter Android app (main mobile application)
│   ├── lib/                 # Dart source code
│   │   ├── pages/           # Full-screen routes (home, camera, details)
│   │   ├── components/      # Widgets scoped by page
│   │   ├── services/        # yolo_detector.dart, pig_detector.dart
│   │   ├── models/          # detection_result.dart
│   │   └── utils/           # camera_metadata, image_preprocessing, pig_math
│   ├── android/             # Android native (Camera2 API, ONNX runtime)
│   ├── assets/              # ONNX models, icons, fonts
│   └── test/                # Unit tests
│
├── pig_classification/      # Breed classification (experimental)
├── pig_segmentation/        # RF-DETR segmentation training notebooks
├── transform/               # Homography / perspective correction notebooks
├── measure_length/          # Pixel-space length calculation notebooks
├── transform_model/         # Weight regression training notebooks
├── depth-estimation/        # Depth model experiments
└── depth/                   # ONNX export & validation scripts
```

---

## Mobile App (DooMoo)

### Setup

```bash
# Install Flutter (see https://flutter.dev)
cd DooMoo
flutter pub get
flutter run
```

### Build Release APK

```bash
cd DooMoo
flutter clean
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

### Build App Bundle (Play Store)

```bash
cd DooMoo
flutter clean
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

### Run Tests

```bash
cd DooMoo
flutter test
flutter test test/widget_test.dart  # single file
flutter analyze                      # static analysis
```

### Pre-commit Hook

The repo includes `.githooks/pre-commit` that auto-runs `dart format .` before every commit. Enable it:

```bash
git config core.hooksPath .githooks
```

---

## App Architecture

### Navigation Flow

```
MyApp
└── HomePage
    ├── Camera button → CameraPage → DetailsPage
    └── Upload button → image_picker → DetailsPage
```

`CameraPage` captures a photo, extracts camera metadata (EXIF + native sensor info), runs YOLOv8 detection, then navigates to `DetailsPage`. RF-DETR segmentation runs **lazily** on `DetailsPage` when the user taps a detected pig.

### Pig Detection Pipeline

**Stage 1 — YOLOv8** (`lib/services/yolo_detector.dart`)
- Model: `assets/models/yolov8_pig.onnx`
- Runs on full image → bounding boxes with confidence
- Filters by confidence ≥ 0.5

**Stage 2 — RF-DETR** (`lib/services/pig_detector.dart`)
- Model: `assets/models/rf_detr_pig.onnx` (input: `[1, 3, 432, 432]`)
- Crops image to YOLO bounding box + 10% padding
- Preprocesses: resize to 432×432, ImageNet normalization, NCHW format
- Outputs: bounding boxes, class logits, segmentation masks
- Filters by confidence ≥ 0.5, attaches mask to detection

### Weight Estimation

`PigInfo` widget (`lib/components/DetailsPage/PigInfo.dart`):
1. User selects a detected pig
2. User inputs camera-to-pig distance in meters
3. Converts pixel to real-world size via pinhole camera model:
   - `real_size = (pixel_size × sensor_size / image_size × distance) / focal_length`
4. Outputs measurements in cm
5. Estimates weight via regression:

```
Weight(kg) = -21.95 + 0.31(Body Length) + 0.43(Chest Width)
           + 0.48(Abdominal Width) + 0.43(Hip Width)
```

All in cm.

### Camera Metadata Extraction

`CameraMetadataExtractor` (`lib/utils/camera_metadata.dart`):
1. Decodes image → pixel dimensions
2. Parses EXIF tags (focal length, f-number, ISO, sensor size)
3. Falls back to native platform channel `"camera_info"` for hardware sensor size
   - Android: Camera2 API `SENSOR_INFO_PHYSICAL_SIZE`
   - iOS: AVFoundation

`CameraMetadataCache` persists hardware metadata to avoid repeated native calls.

---

## Model Training Notebooks

| Directory | Purpose |
|---|---|
| `pig_segmentation/` | RF-DETR segmentation training |
| `transform/` | Homography / perspective correction |
| `measure_length/` | Pixel-space length calculations |
| `transform_model/` | Weight regression model training |
| `depth-estimation/` | Depth model experiments & validation |
| `pig_classification/` | Breed classification (experimental) |

---

## Technical Specs

| Constraint | Detail |
|---|---|
| Platform | Android only |
| Connectivity | Fully offline — no network calls |
| Device class | Helio G85 / Snapdragon 680+ |
| RAM | 3–4 GB typical |
| Inference budget | <3 seconds total |
| Camera view | Top-down only |
| ML runtime | ONNX (via `onnxruntime_v2`) |
| Detection model | YOLOv8 (ONNX) |
| Segmentation model | RF-DETR (ONNX, 432×432 input) |
| Weight model | Linear regression |

---

## CI Pipeline

Defined in `.github/workflows/flutter_ci.yml`:
- Runs on push/PR to `main`
- `flutter pub get`
- `dart format --set-exit-if-changed .`
- `flutter test --coverage`

---

## Related

- [Project README](./README.md) — project overview & problem context
- `DooMoo/CLAUDE.md` — detailed Claude Code instructions
- `model.ipynb` — full Python pipeline reference
