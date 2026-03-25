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

  /// The coordinate region in the original image that the [mask] pixels correspond to.
  /// If null, mask is assumed to cover the entire image.
  final Rect? maskRect;

  const PigDetection({
    required this.boundingBox,
    required this.confidence,
    required this.classId,
    this.mask,
    this.maskRect,
  });

  PigDetection copyWith({
    Rect? boundingBox,
    double? confidence,
    int? classId,
    List<List<double>>? mask,
    Rect? maskRect,
  }) {
    return PigDetection(
      boundingBox: boundingBox ?? this.boundingBox,
      confidence: confidence ?? this.confidence,
      classId: classId ?? this.classId,
      mask: mask ?? this.mask,
      maskRect: maskRect ?? this.maskRect,
    );
  }

  @override
  String toString() =>
      'PigDetection(box: $boundingBox, conf: ${confidence.toStringAsFixed(2)}, cls: $classId, hasMask: ${mask != null})';
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

  DetectionResult copyWith({
    List<PigDetection>? detections,
    int? imageWidth,
    int? imageHeight,
  }) {
    return DetectionResult(
      detections: detections ?? this.detections,
      imageWidth: imageWidth ?? this.imageWidth,
      imageHeight: imageHeight ?? this.imageHeight,
    );
  }

  bool get isEmpty => detections.isEmpty;
  int get count => detections.length;

  @override
  String toString() =>
      'DetectionResult(count: $count, imgSize: ${imageWidth}x$imageHeight)';
}
