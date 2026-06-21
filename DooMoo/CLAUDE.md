# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DooMoo** (BlackPig) is a Flutter mobile application for capturing and analyzing pig images. It runs on-device YOLOv8 pig detection + RF-DETR segmentation (both ONNX), extracts camera metadata (EXIF + native hardware info), converts pixel measurements to real-world sizes, and estimates pig weight using a regression model. The app targets Android and iOS primarily.

## Common Commands

```bash
# Install dependencies
flutter pub get

# Run on connected device
flutter run

# Run on specific emulator
flutter run -d emulator-5554

# Run tests
flutter test

# Run a single test file
flutter test test/widget_test.dart

# Static analysis / lint
flutter analyze

# Build release APK
flutter clean && flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk

# Build Android App Bundle (for Play Store)
flutter clean && flutter build appbundle --release
```

## Architecture

### Navigation Flow

```
MyApp
└── HomePage
    ├── Camera button → CameraPage → DetailsPage
    └── Upload button → image_picker → DetailsPage
```

`CameraPage` uses `image_picker` to capture a photo, then extracts `CameraMetadata` and runs YOLOv8 detection before navigating to `DetailsPage`. RF-DETR segmentation runs lazily on `DetailsPage` when the user taps a detected pig. The Upload path skips `CameraPage` and calls the metadata extractor + YOLO detector directly on the picked file.

### Key Directories

- `lib/pages/` — Full-screen routes: `home.dart`, `camera.dart`, `details.dart`
- `lib/components/` — Widget components scoped by page (`HomePage/`, `DetailsPage/`)
- `lib/services/` — `yolo_detector.dart` (YOLOv8 ONNX detection), `pig_detector.dart` (RF-DETR ONNX segmentation)
- `lib/models/` — `detection_result.dart` (PigDetection, DetectionResult)
- `lib/utils/` — Shared utilities: `camera_metadata.dart` (EXIF + platform channel), `image_preprocessing.dart`, `responsive.dart`

### Pig Detection Pipeline

Two-stage pipeline:

**Stage 1 — YOLOv8** (`YoloDetector` in `lib/services/yolo_detector.dart`):
1. Loads ONNX model: `assets/models/yolov8_pig.onnx`
2. Runs inference on full image → bounding boxes with confidence
3. Filters by confidence (0.5), returns `List<PigDetection>` with bounding boxes (no masks)

**Stage 2 — RF-DETR** (`PigDetector` in `lib/services/pig_detector.dart`):
1. Runs lazily when user taps a pig on `DetailsPage`
2. Loads ONNX model: `assets/models/rf_detr_pig.onnx` (input: `[1, 3, 432, 432]`)
3. Crops image to YOLO's bounding box (with 10% padding)
4. Preprocesses: resize to 432x432, normalize with ImageNet stats, NCHW format
5. Runs inference → outputs: bounding boxes, class logits, segmentation masks
6. Postprocesses: filter by confidence (0.5), extracts segmentation mask
7. Returns `PigDetection` with mask attached

### Size Calculation & Weight Estimation Pipeline

`PigInfo` (in `lib/components/DetailsPage/PigInfo.dart`) is a StatefulWidget that:
1. User selects a detected pig by tapping its bounding box
2. User inputs camera-to-pig distance in **meters**
3. Converts pixel measurements to real-world size using pinhole camera model:
   - Horizontal: `real_size = (pixel_width × sensorWidth / imageWidth × distance) / focalLength`
   - Vertical: `real_size = (pixel_height × sensorHeight / imageHeight × distance) / focalLength`
4. Outputs measurements in **cm**
5. Estimates weight using regression model:
   ```
   Weight(kg) = -21.95431 + 0.31079(Body Length cm) + 0.43166(Chest Width cm)
              + 0.47990(Abdominal Width cm) + 0.42656(Hip Width cm)
   ```
   Note: Currently Chest/Abdominal/Hip widths all use bounding box height (single width measurement).

### Camera Metadata Pipeline

`CameraMetadataExtractor.extractFromImage(filePath)` (in `lib/utils/camera_metadata.dart`):
1. Decodes the image to get pixel dimensions
2. Parses EXIF tags (focal length, f-number, ISO, sensor dimensions)
3. Falls back to native platform channel `"camera_info"` for hardware sensor size
   - Android: `android/app/src/main/kotlin/com/example/blackpig/MainActivity.kt` (Camera2 API `SENSOR_INFO_PHYSICAL_SIZE`)
   - iOS: `ios/Runner/AppDelegate.swift` (AVFoundation)
4. Returns a `CameraMetadata` object passed via route arguments to `DetailsPage`

`CameraMetadataCache` persists hardware metadata to the app documents directory to avoid repeated native calls.

### Responsive Design

All sizes go through `ResponsiveUtils` (`lib/utils/responsive.dart`), which scales dimensions as a percentage of screen width/height relative to a 375 px base (iPhone 6/7/8). Use `ResponsiveUtils.wp(x)` / `ResponsiveUtils.hp(x)` for widths/heights.

### State Management

No external state management library. Pages use minimal `StatefulWidget`. Data is passed directly as constructor arguments through `Navigator.push`.

## Tech Stack

| Area | Library |
|---|---|
| Image pick / camera | `image_picker ^1.2.0` |
| EXIF parsing | `exif ^3.3.0` |
| ONNX inference | `onnxruntime ^1.4.1` |
| Image processing | `image ^4.3.0` |
| SVG rendering | `flutter_svg ^2.2.0` |
| File paths | `path_provider ^2.1.2` |
| Fonts | DB HelvethaicaMon X (Thai) |

## Related Files

- `model.ipynb` (in parent directory `/Users/nine/Desktop/CMKL/2.2/urd/`) — Jupyter notebook with the full Python pipeline (YOLO, RF-DETR, Depth Pro, PCA measurements, size calculation). The Flutter app replicates this pipeline on-device.

## Current Limitations

- Chest, Abdominal, and Hip widths all use the same bounding box height value (no PCA-based distinct width measurements yet).
- iOS sensor physical size returns `0.0` (AVFoundation lookup incomplete).
- No on-device depth model — distance must be manually input by the user.
