import 'package:DooMoo/models/breed_dimensions.dart';
import 'package:DooMoo/utils/camera_metadata.dart';

class DepthEstimateResult {
  final double distanceMeters;
  final double confidence;

  const DepthEstimateResult({
    required this.distanceMeters,
    required this.confidence,
  });
}

class DepthService {
  /// Estimate camera-to-pig distance using the pinhole camera model.
  ///
  /// Formula (similar triangles):
  ///   distance_mm = (real_pig_length_mm × focal_length_px) / pixel_pig_length
  ///
  /// Where:
  ///   focal_length_px = (focal_length_mm / sensor_width_mm) × image_width_px
  ///
  /// Returns null if any required value is missing or invalid.
  static DepthEstimateResult? estimateGeometric({
    required double pixelPigLength,
    required CameraMetadata? cameraMetadata,
    String breedLabel = 'unknown',
  }) {
    final meta = cameraMetadata;
    if (meta == null) return null;

    final focalMm = meta.focalLength;
    final sensorW = meta.sensorWidth;
    final sensorH = meta.sensorHeight;
    final imgW = meta.imageWidth;
    final imgH = meta.imageHeight;

    if (focalMm == null || focalMm <= 0) return null;
    if (sensorW == null || sensorW <= 0) return null;
    if (sensorH == null || sensorH <= 0) return null;
    if (imgW == null || imgW <= 0) return null;
    if (imgH == null || imgH <= 0) return null;
    if (pixelPigLength <= 0) return null;

    // Convert focal length from mm to pixels using sensor size.
    // Average horizontal + vertical to match pig_math.dart pixel-size logic.
    final focalPxH = (focalMm / sensorW) * imgW;
    final focalPxV = (focalMm / sensorH) * imgH;
    final focalPx = (focalPxH + focalPxV) / 2;

    // Real pig body length from breed lookup
    final realLengthMm = breedRealLengthFor(breedLabel);

    // Pinhole camera formula
    final distanceMm = (realLengthMm * focalPx) / pixelPigLength;
    final distanceMeters = distanceMm / 1000.0;

    if (distanceMeters <= 0 || distanceMeters > 10.0) return null;

    // Confidence: lower when breed is unknown (real length is just an average)
    final confidence = breedLabel == 'unknown' ? 0.6 : 0.85;

    return DepthEstimateResult(
      distanceMeters: distanceMeters,
      confidence: confidence,
    );
  }
}
