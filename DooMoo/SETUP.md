# DooMoo — Setup & Run Guide

> Flutter app for the Black Pig Project. Detects pigs from a top-down camera image and estimates weight using an on-device ONNX model.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Flutter SDK | ≥ 3.6.2 | https://docs.flutter.dev/get-started/install |
| Dart SDK | ≥ 3.6.2 | Bundled with Flutter |
| Android Studio | Any recent | For Android emulator / device |
| Xcode | ≥ 14 (macOS only) | For iOS builds |

Verify your setup:
```bash
flutter doctor
```
All checkmarks should be green before continuing.

---

## 1. Get the ONNX Model

The model file is **not included in the repo** (too large for git).
Download `rf_detr_pig.onnx` and place it at:

```
DooMoo/assets/models/rf_detr_pig.onnx
```

> Ask the project team for the model file, or check the project's shared storage (Google Drive / release page).

---

## 2. Install Dependencies

```bash
cd DooMoo
flutter pub get
```

---

## 3. Run the App

### On a physical Android device
1. Enable **Developer Options** and **USB Debugging** on your phone
2. Connect via USB
3. Run:
```bash
flutter run
```

### On an Android emulator
1. Open Android Studio → Virtual Device Manager → start an emulator
2. Run:
```bash
flutter run
```

### On iOS (macOS only)
```bash
cd ios
pod install
cd ..
flutter run
```

---

## 4. Build a Release APK

```bash
flutter clean
flutter build apk --release
```

Output: `build/app/outputs/flutter-apk/app-release.apk`
Transfer the APK to your Android device and install it directly.

---

## App Structure

```
lib/
├── main.dart           # Entry point
├── pages/
│   ├── home.dart       # Home screen
│   ├── camera.dart     # Camera capture page
│   ├── details.dart    # Results / measurement page
│   └── pig_detector.dart
├── components/         # Reusable UI widgets
├── models/             # Data models
├── services/           # ONNX inference service
└── utils/
    └── camera_metadata.dart  # EXIF / sensor data extraction
```

---

## Troubleshooting

**`flutter pub get` fails**
Make sure Flutter is in your PATH: `export PATH="$PATH:/path/to/flutter/bin"`

**Model not found at runtime**
Double-check the file is at `assets/models/rf_detr_pig.onnx` and that `pubspec.yaml` lists it under `assets`.

**Camera permission denied**
On Android, go to Settings → Apps → blackpig → Permissions → allow Camera and Storage.

**`pod install` fails on iOS**
Run `sudo gem install cocoapods` then retry.
