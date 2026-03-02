# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DooMoo** (BlackPig) is a Flutter mobile application for capturing and analyzing pig images. It extracts camera metadata (EXIF + native hardware info) and displays pig measurements. The app targets Android and iOS primarily, with web/desktop platform files present.

## Common Commands

```bash
# Install dependencies
flutter pub get

# Run on connected device
flutter run

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

`CameraPage` uses `image_picker` to capture a photo, then extracts `CameraMetadata` before navigating to `DetailsPage`. The Upload path skips `CameraPage` and calls the metadata extractor directly on the picked file.

### Key Directories

- `lib/pages/` — Full-screen routes: `home.dart`, `camera.dart`, `details.dart`
- `lib/components/` — Widget components scoped by page (`HomePage/`, `DetailsPage/`)
- `lib/utils/` — Shared utilities: `camera_metadata.dart` (EXIF + platform channel), `responsive.dart` (screen-relative sizing)

### Camera Metadata Pipeline

`CameraMetadataExtractor.extractFromImage(filePath)` (in `lib/utils/camera_metadata.dart`):
1. Decodes the image to get pixel dimensions
2. Parses EXIF tags (focal length, f-number, ISO, sensor dimensions)
3. Falls back to native platform channel `"camera_info"` for hardware sensor size
   - Android: `android/app/src/main/kotlin/com/example/blackpig/MainActivity.kt` (Camera2 API)
   - iOS: `ios/Runner/AppDelegate.swift` (AVFoundation)
4. Returns a `CameraMetadata` object passed via route arguments to `DetailsPage`

`CameraMetadataCache` persists hardware metadata to the app documents directory to avoid repeated native calls.

### Responsive Design

All sizes go through `ResponsiveUtils` (`lib/utils/responsive.dart`), which scales dimensions as a percentage of screen width/height relative to a 375 px base (iPhone 6/7/8). Use `ResponsiveUtils.wp(x)` / `ResponsiveUtils.hp(x)` for widths/heights.

### State Management

No external state management library. Pages are stateless or use minimal `StatefulWidget` (e.g., `PigImage` for aspect ratio). Data is passed directly as constructor arguments through `Navigator.push`.

## Tech Stack

| Area | Library |
|---|---|
| Image pick / camera | `image_picker ^1.2.0` |
| EXIF parsing | `exif ^3.3.0` |
| SVG rendering | `flutter_svg ^2.2.0` |
| File paths | `path_provider ^2.1.2` |
| Fonts | DB HelvethaicaMon X (Thai) |

## Current State

- `PigInfo` component (`lib/components/DetailsPage/PigInfo.dart`) is a UI placeholder — no actual measurement logic yet.
- iOS sensor physical size returns `0.0` (AVFoundation lookup incomplete).
- No ML model integration; pig analysis is a planned next step.
