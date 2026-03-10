import 'dart:io';
import 'dart:typed_data';
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
    final channelSize = targetSize * targetSize;
    final data = Float32List(1 * 3 * channelSize);

    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        final idx = y * targetSize + x;

        // Normalize: (pixel / 255 - mean) / std
        data[0 * channelSize + idx] = (pixel.r / 255.0 - _mean[0]) / _std[0]; // R
        data[1 * channelSize + idx] = (pixel.g / 255.0 - _mean[1]) / _std[1]; // G
        data[2 * channelSize + idx] = (pixel.b / 255.0 - _mean[2]) / _std[2]; // B
      }
    }

    return PreprocessedImage(
      data: data,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
    );
  }
}

class PreprocessedImage {
  final Float32List data;
  final int originalWidth;
  final int originalHeight;

  const PreprocessedImage({
    required this.data,
    required this.originalWidth,
    required this.originalHeight,
  });
}
