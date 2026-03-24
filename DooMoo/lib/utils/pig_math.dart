class PigMath {
  /// Convert pixel length to real-world mm using camera metadata and distance.
  ///
  /// [pixelLength] the length in pixels
  /// [distanceMm] the distance from camera to object in mm
  /// [focalLength] camera focal length in mm
  /// [sensorWidth] camera sensor width in mm
  /// [sensorHeight] camera sensor height in mm
  /// [imageWidth] image width in pixels
  /// [imageHeight] image height in pixels
  static double? pixelToMm({
    required double? pixelLength,
    required double? distanceMm,
    required double? focalLength,
    required double? sensorWidth,
    required double? sensorHeight,
    required int? imageWidth,
    required int? imageHeight,
  }) {
    if (pixelLength == null ||
        distanceMm == null ||
        focalLength == null ||
        sensorWidth == null ||
        sensorHeight == null ||
        imageWidth == null ||
        imageHeight == null) return null;

    if (focalLength <= 0 ||
        sensorWidth <= 0 ||
        sensorHeight <= 0 ||
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        distanceMm <= 0) return null;

    // Average pixel size since PCA axes are not aligned to image axes
    final pixelSizeW = sensorWidth / imageWidth;
    final pixelSizeH = sensorHeight / imageHeight;
    final pixelSizeMm = (pixelSizeW + pixelSizeH) / 2;

    final objectOnSensorMm = pixelLength * pixelSizeMm;
    return (objectOnSensorMm * distanceMm) / focalLength;
  }

  /// Estimate weight using regression model:
  /// Weight = -21.95431 + 0.31079(Body Length) + 0.43166(Chest Width)
  ///        + 0.47990(Abdominal Width) + 0.42656(Hip Width)
  /// All inputs in mm (converted to cm internally), output in kg.
  static String estimateWeight({
    required double? bodyLengthMm,
    required double? chestWidthMm,
    required double? abdominalWidthMm,
    required double? hipWidthMm,
  }) {
    if (bodyLengthMm == null ||
        chestWidthMm == null ||
        abdominalWidthMm == null ||
        hipWidthMm == null) return '-';
    final weight = -21.95431
        + 0.31079 * (bodyLengthMm / 10)
        + 0.43166 * (chestWidthMm / 10)
        + 0.47990 * (abdominalWidthMm / 10)
        + 0.42656 * (hipWidthMm / 10);
    if (weight < 0) return '-';
    return '${weight.toStringAsFixed(1)} kg';
  }
}
