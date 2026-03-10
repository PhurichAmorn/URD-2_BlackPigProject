import 'dart:ui';

class PigDetection {
  /// Bounding box in xyxy format (left, top, right, bottom) in original image pixels.
  final Rect boundingBox;

  /// Confidence score (0.0 - 1.0).
  final double confidence;

  /// Class ID from the model.
  final int classId;

  /// Pixel-level segmentation mask sized to [maskHeight x maskWidth].
  /// Values are 0.0 - 1.0 (probability). Null if segmentation unavailable.
  final List<List<double>>? mask;

  const PigDetection({
    required this.boundingBox,
    required this.confidence,
    required this.classId,
    this.mask,
  });

  @override
  String toString() =>
      'PigDetection(box: $boundingBox, conf: ${confidence.toStringAsFixed(2)}, cls: $classId)';
}

class DetectionResult {
  /// Detected pigs.
  final List<PigDetection> detections;

  /// Original image dimensions used for coordinate mapping.
  final int imageWidth;
  final int imageHeight;

  const DetectionResult({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  bool get isEmpty => detections.isEmpty;
  int get count => detections.length;

  @override
  String toString() =>
      'DetectionResult(count: $count, imgSize: ${imageWidth}x$imageHeight)';
}
