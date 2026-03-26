import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:DooMoo/models/detection_result.dart';
import 'package:DooMoo/utils/image_preprocessing.dart';

/// Singleton service for running RF-DETR pig segmentation inference on-device.
class PigDetector {
  static PigDetector? _instance;
  static Future<PigDetector>? _initFuture;
  static bool _envInitialized = false;

  OrtSession? _session;
  bool _isReady = false;

  // ---- Model constants ----
  static const String _modelAsset = 'assets/models/rf_detr_pig.onnx';
  static const int inputSize = 432; // From ONNX model: input [1, 3, 432, 432]
  static const String _inputName = 'input';
  static const double _confidenceThreshold = 0.5;
  static const double _maskThreshold = 0.0;
  static const int _maxDetections = 10;

  PigDetector._();

  /// Returns the singleton instance, loading the model on first call.
  static Future<PigDetector> getInstance() async {
    if (_instance != null && _instance!._isReady) return _instance!;

    if (_initFuture != null) return _initFuture!;

    _initFuture = _initInstance();
    return _initFuture!;
  }

  static Future<PigDetector> _initInstance() async {
    try {
      final detector = PigDetector._();
      await detector._loadModel();
      _instance = detector;
      return detector;
    } catch (e) {
      _initFuture = null;
      rethrow;
    }
  }

  Future<void> _loadModel() async {
    if (!_envInitialized) {
      OrtEnv.instance.init();
      _envInitialized = true;
    }

    final rawAssetData = await rootBundle.load(_modelAsset);
    final modelBytes = rawAssetData.buffer.asUint8List();

    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(modelBytes, sessionOptions);
    sessionOptions.release();

    _isReady = true;
  }

  /// Run detection/segmentation on an image file.
  /// If [cropRect] is provided, it crops the image first before feeding to RF-DETR.
  Future<DetectionResult> detect(String imagePath, {Rect? cropRect}) async {
    if (!_isReady || _session == null) {
      await _loadModel();
      if (!_isReady || _session == null) {
        throw StateError('PigDetector not ready after manual load attempt');
      }
    }

    // 1. Preprocess
    final preprocessed = cropRect != null
        ? ImagePreprocessor.cropAndPreprocess(imagePath, cropRect, inputSize)
        : ImagePreprocessor.preprocess(imagePath, inputSize);

    // 2. Create input tensor [1, 3, H, W]
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      preprocessed.data,
      [1, 3, inputSize, inputSize],
    );

    final runOptions = OrtRunOptions();

    // 3. Run inference
    final outputs = await _session!.runAsync(
      runOptions,
      {_inputName: inputTensor},
    );

    final outputList = outputs ?? [];

    // 4. Postprocess outputs
    final detections = _postprocess(
      outputList,
      preprocessed,
    );

    // 5. Cleanup
    inputTensor.release();
    runOptions.release();
    for (final output in outputList) {
      output?.release();
    }

    return DetectionResult(
      detections: detections,
      imageWidth: preprocessed.originalWidth,
      imageHeight: preprocessed.originalHeight,
    );
  }

  List<PigDetection> _postprocess(
    List<OrtValue?> outputs,
    PreprocessedImage preprocessed,
  ) {
    if (outputs.isEmpty || outputs[0] == null || outputs[1] == null) {
      return [];
    }

    final originalWidth = preprocessed.originalWidth;
    final originalHeight = preprocessed.originalHeight;
    final cropRect = preprocessed.cropRect;

    final detsRaw = outputs[0]!.value;
    final labelsRaw = outputs[1]!.value;
    final masksRaw = outputs.length > 2 ? outputs[2]?.value : null;

    final List<List<double>> dets = _parseDets(detsRaw);
    final List<double> scores = _parseLabels(labelsRaw);
    final int numDetections = dets.length;

    final List<PigDetection> detections = [];

    // Determine if coords are normalized [0,1] or in input pixel space [0, inputSize]
    bool isNormalized = true;
    for (int i = 0; i < math.min(10, dets.length); i++) {
      if (dets[i].any((v) => v.abs() > 1.5)) {
        isNormalized = false;
        break;
      }
    }

    // Scale factors from input space to the CURRENT input space (which might be a crop)
    final double currentWidth = cropRect?.width ?? originalWidth.toDouble();
    final double currentHeight = cropRect?.height ?? originalHeight.toDouble();
    final double offsetX = cropRect?.left ?? 0;
    final double offsetY = cropRect?.top ?? 0;

    final double scaleToCurrentX = currentWidth / inputSize;
    final double scaleToCurrentY = currentHeight / inputSize;

    for (int i = 0;
        i < numDetections && detections.length < _maxDetections;
        i++) {
      final score = scores[i];
      if (score < _confidenceThreshold) continue;

      double cx, cy, w, h;
      if (isNormalized) {
        // Normalized [0,1] within the current input (crop or full)
        cx = dets[i][0] * currentWidth + offsetX;
        cy = dets[i][1] * currentHeight + offsetY;
        w = dets[i][2] * currentWidth;
        h = dets[i][3] * currentHeight;
      } else {
        // Input pixel space [0, inputSize] — scale to original image pixels
        cx = dets[i][0] * scaleToCurrentX + offsetX;
        cy = dets[i][1] * scaleToCurrentY + offsetY;
        w = dets[i][2] * scaleToCurrentX;
        h = dets[i][3] * scaleToCurrentY;
      }

      final left = (cx - w / 2).clamp(0.0, originalWidth.toDouble());
      final top = (cy - h / 2).clamp(0.0, originalHeight.toDouble());
      final right = (cx + w / 2).clamp(0.0, originalWidth.toDouble());
      final bottom = (cy + h / 2).clamp(0.0, originalHeight.toDouble());

      // Parse mask for this detection if available
      List<List<double>>? mask;
      if (masksRaw != null) {
        mask = _parseMask(masksRaw, i, preprocessed);
      }

      detections.add(PigDetection(
        boundingBox: Rect.fromLTRB(left, top, right, bottom),
        confidence: score,
        classId: 0,
        mask: mask,
        maskRect: cropRect ??
            Rect.fromLTRB(
                0, 0, originalWidth.toDouble(), originalHeight.toDouble()),
      ));
    }

    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    return detections;
  }

  /// Parse detection boxes from model output. Handles nested list structures.
  List<List<double>> _parseDets(dynamic raw) {
    // Expected: [[[cx, cy, w, h], ...]] → shape [1, N, 4]
    if (raw is List && raw.isNotEmpty && raw[0] is List) {
      final batch = raw[0] as List;
      return batch.map<List<double>>((det) {
        if (det is List) {
          return det.map<double>((v) => (v as num).toDouble()).toList();
        }
        return <double>[];
      }).toList();
    }

    // Flat float array: reshape [1, N, 4]
    if (raw is Float32List ||
        (raw is List && raw.isNotEmpty && raw[0] is num)) {
      final flat = raw is Float32List
          ? raw
          : Float32List.fromList((raw as List).cast<double>());
      final n = flat.length ~/ 4;
      return List.generate(
          n,
          (i) =>
              [flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2], flat[i * 4 + 3]]);
    }

    return [];
  }

  /// Parse confidence scores from labels output.
  List<double> _parseLabels(dynamic raw) {
    // Could be [1, N] nested or [1, N, num_classes] — take max across classes via sigmoid
    if (raw is List && raw.isNotEmpty && raw[0] is List) {
      final batch = raw[0] as List;
      return batch.map<double>((v) {
        if (v is num) return _sigmoid(v.toDouble());
        if (v is List) {
          // Multi-class: take max score
          double maxVal = double.negativeInfinity;
          for (final s in v) {
            final sv = (s as num).toDouble();
            if (sv > maxVal) maxVal = sv;
          }
          return _sigmoid(maxVal);
        }
        return 0.0;
      }).toList();
    }

    if (raw is Float32List ||
        (raw is List && raw.isNotEmpty && raw[0] is num)) {
      final flat = raw is Float32List
          ? raw
          : Float32List.fromList((raw as List).cast<double>());
      return flat.map((v) => _sigmoid(v)).toList();
    }

    return [];
  }

  /// Parse a single detection's mask from the masks output.
  List<List<double>>? _parseMask(
    dynamic masksRaw,
    int detectionIndex,
    PreprocessedImage preprocessed,
  ) {
    try {
      // Expected shape: [1, N, maskH, maskW]
      dynamic maskData;
      if (masksRaw is List && masksRaw.isNotEmpty && masksRaw[0] is List) {
        final batch = masksRaw[0] as List;
        if (detectionIndex < batch.length) {
          maskData = batch[detectionIndex];
        }
      }

      if (maskData == null) return null;

      // maskData should be [maskH][maskW]
      if (maskData is List && maskData.isNotEmpty && maskData[0] is List) {
        return maskData.map<List<double>>((row) {
          return (row as List).map<double>((v) {
            final val = (v as num).toDouble();
            return val > _maskThreshold ? val : 0.0;
          }).toList();
        }).toList();
      }
    } catch (_) {
      // Mask parsing failed
    }
    return null;
  }

  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  /// Release the ONNX session and free resources.
  void dispose() {
    _session?.release();
    _session = null;
    _isReady = false;
    _instance = null;
    _initFuture = null;
  }
}
