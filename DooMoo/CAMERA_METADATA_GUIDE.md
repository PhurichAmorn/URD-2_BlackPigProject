# Camera Metadata Extraction Guide

This guide explains how to extract camera sensor dimensions (sensor_width, sensor_height) and focal length from photos taken with the camera.

## Overview

The app now includes functionality to extract camera metadata from captured images. This includes:
- **Sensor Width** (in mm)
- **Sensor Height** (in mm)  
- **Focal Length** (in mm)
- **Image Dimensions** (width x height in pixels)
- **F-Number** (aperture)
- **ISO** sensitivity

## How It Works

### Method 1: EXIF Data Extraction (Primary)
When you capture or upload an image, the app automatically extracts EXIF metadata embedded in the image file. This includes focal length and other camera settings.

### Method 2: Hardware Access (Fallback)
If EXIF data is not available, the app attempts to get camera hardware characteristics directly from the device using platform channels.

## Usage

### In Your Code

The metadata is automatically extracted when you capture or upload an image. You can access it in the `DetailsPage`:

```dart
// In camera.dart or Upload.dart
final metadata = await CameraMetadataExtractor.extractFromImage(xfile.path);

// Pass to DetailsPage
DetailsPage(
  imagePath: xfile.path,
  cameraMetadata: metadata,
)
```

### Accessing Metadata Values

```dart
// In DetailsPage or any component
final metadata = widget.cameraMetadata;

if (metadata != null) {
  final sensorWidth = metadata.sensorWidth;    // in mm
  final sensorHeight = metadata.sensorHeight;  // in mm
  final focalLength = metadata.focalLength;     // in mm
  final imageWidth = metadata.imageWidth;      // in pixels
  final imageHeight = metadata.imageHeight;    // in pixels
  final fNumber = metadata.fNumber;            // aperture
  final iso = metadata.iso;                     // ISO value
}
```

## Platform-Specific Implementation

### Android
The Android implementation uses the Camera2 API to get hardware characteristics:
- Sensor physical size (width/height in mm)
- Available focal lengths

**Location:** `android/app/src/main/kotlin/com/example/blackpig/MainActivity.kt`

### iOS
The iOS implementation uses AVFoundation:
- Focal length from lens position
- Pixel dimensions from active format

**Note:** iOS doesn't directly provide physical sensor size. You may need to add a device database for accurate sensor dimensions.

**Location:** `ios/Runner/AppDelegate.swift`

## Limitations

1. **EXIF Data Availability**: Not all images contain complete EXIF metadata. Some images may be stripped of metadata for privacy reasons.

2. **Sensor Dimensions**: 
   - Android: Available directly from Camera2 API
   - iOS: Requires device-specific lookup (not implemented by default)

3. **Focal Length**: Usually available in EXIF data, but may vary by device and camera app.

## Example Output

When you capture a photo, you'll see debug output like:
```
Camera Metadata: CameraMetadata(
  sensorWidth: 4.8 mm, 
  sensorHeight: 3.6 mm, 
  focalLength: 3.99 mm, 
  imageSize: 2048x1536, 
  fNumber: 2.2, 
  iso: 100
)
```

## Troubleshooting

### No Metadata Available
If metadata is null or empty:
1. Check if the image has EXIF data (some apps strip metadata)
2. Verify camera permissions are granted
3. Check platform channel implementation is correct

### Sensor Dimensions Missing
- On Android: Should work automatically
- On iOS: You may need to add a device database mapping device models to sensor sizes

## Next Steps

To use this metadata for pig measurement calculations:
1. Access `cameraMetadata` in your `PigInfo` component
2. Use sensor dimensions and focal length to calculate real-world measurements
3. Combine with image analysis results from your ML model

## Files Modified

- `lib/utils/camera_metadata.dart` - Main metadata extraction utility
- `lib/pages/camera.dart` - Extracts metadata on capture
- `lib/components/HomePage/Upload.dart` - Extracts metadata on upload
- `lib/pages/details.dart` - Receives and can use metadata
- `android/app/src/main/kotlin/com/example/blackpig/MainActivity.kt` - Android platform channel
- `ios/Runner/AppDelegate.swift` - iOS platform channel
- `pubspec.yaml` - Added `exif` package dependency

