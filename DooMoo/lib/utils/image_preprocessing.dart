import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;

/// Preprocesses images for the RF-DETR ONNX model.
class ImagePreprocessor {
  // ImageNet normalization constants
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  /// Decode, resize, normalize an image file for model input.
  ///
  /// Returns a [Float32List] shaped [1, 3, height, width] (NCHW).
  /// Also returns the original image dimensions via [originalSize].
  static PreprocessedImage preprocess(String imagePath, int targetSize) {
    final bytes = File(imagePath).readAsBytesSync();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode image: $imagePath');
    }

    final originalWidth = decoded.width;
    final originalHeight = decoded.height;

    // Resize with bilinear interpolation
    final resized = img.copyResize(
      decoded,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.linear,
    );

    // Convert to NCHW float32 with ImageNet normalization
    final data = _normalize(resized, targetSize);

    return PreprocessedImage(
      data: data,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      targetWidth: targetSize,
      targetHeight: targetSize,
    );
  }

  /// Crop, resize, and normalize an image for model input.
  static PreprocessedImage cropAndPreprocess(
    String imagePath,
    Rect cropRect,
    int targetSize,
  ) {
    final bytes = File(imagePath).readAsBytesSync();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode image: $imagePath');
    }

    final originalWidth = decoded.width;
    final originalHeight = decoded.height;

    // 1. Crop the image
    final cropX = cropRect.left.toInt().clamp(0, originalWidth - 1);
    final cropY = cropRect.top.toInt().clamp(0, originalHeight - 1);
    final cropW = cropRect.width.toInt().clamp(1, originalWidth - cropX);
    final cropH = cropRect.height.toInt().clamp(1, originalHeight - cropY);

    final cropped = img.copyCrop(
      decoded,
      x: cropX,
      y: cropY,
      width: cropW,
      height: cropH,
    );

    // 2. Resize
    final resized = img.copyResize(
      cropped,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.linear,
    );

    // 3. Normalize
    final data = _normalize(resized, targetSize);

    return PreprocessedImage(
      data: data,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      cropRect: cropRect,
      targetWidth: targetSize,
      targetHeight: targetSize,
    );
  }

  static Float32List _normalize(img.Image image, int size) {
    final channelSize = size * size;
    final data = Float32List(1 * 3 * channelSize);

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final pixel = image.getPixel(x, y);
        final idx = y * size + x;

        // Normalize: (pixel / 255 - mean) / std
        data[0 * channelSize + idx] = (pixel.r / 255.0 - _mean[0]) / _std[0]; // R
        data[1 * channelSize + idx] = (pixel.g / 255.0 - _mean[1]) / _std[1]; // G
        data[2 * channelSize + idx] = (pixel.b / 255.0 - _mean[2]) / _std[2]; // B
      }
    }
    return data;
  }
}

class PreprocessedImage {
  final Float32List data;
  final int originalWidth;
  final int originalHeight;
  final Rect? cropRect;
  final int targetWidth;
  final int targetHeight;

  const PreprocessedImage({
    required this.data,
    required this.originalWidth,
    required this.originalHeight,
    required this.targetWidth,
    required this.targetHeight,
    this.cropRect,
  });
}
