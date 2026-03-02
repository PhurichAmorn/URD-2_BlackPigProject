import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:blackpig/models/detection_result.dart';
import 'package:blackpig/utils/image_preprocessing.dart';

/// Singleton service for running RF-DETR pig segmentation inference on-device.
class PigDetector {
  static PigDetector? _instance;
  static bool _envInitialized = false;

  OrtSession? _session;
  bool _isReady = false;

  // ---- Model constants ----
  // TODO: After exporting with scripts/export_onnx.py, open the .onnx in Netron
  // and update these values to match your model's actual tensor names and shapes.
  static const String _modelAsset = 'assets/models/rf_detr_pig.onnx';
  static const int inputSize = 432; // From ONNX model: input [1, 3, 432, 432]
  static const String _inputName = 'input';
  // Output tensor names/shapes (from Netron/onnx inspection):
  // outputs[0] = dets:  [1, 200, 4]    — cxcywh normalized bboxes
  // outputs[1] = labels: [1, 200, 2]   — 2-class logits (apply sigmoid)
  // outputs[2] = masks: [1, 200, 108, 108] — segmentation masks
  static const double _confidenceThreshold = 0.5;
  static const double _maskThreshold = 0.0;
  static const int _maxDetections = 10;

  PigDetector._();

  /// Returns the singleton instance, loading the model on first call.
  static Future<PigDetector> getInstance() async {
    if (_instance != null && _instance!._isReady) return _instance!;

    _instance = PigDetector._();
    await _instance!._loadModel();
    return _instance!;
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

  /// Run detection on an image file. Returns bounding boxes, confidence, and masks.
  Future<DetectionResult> detect(String imagePath) async {
    if (!_isReady || _session == null) {
      throw StateError('PigDetector not initialized. Call getInstance() first.');
    }

    // 1. Preprocess
    final preprocessed = ImagePreprocessor.preprocess(imagePath, inputSize);

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
      preprocessed.originalWidth,
      preprocessed.originalHeight,
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
    int originalWidth,
    int originalHeight,
  ) {
    // Output indices depend on model export order.
    // Typically: outputs[0] = dets [1, N, 4], outputs[1] = labels [1, N], outputs[2] = masks [1, N, mH, mW]
    // TODO: Verify indices after Netron inspection. Adjust if needed.

    if (outputs.isEmpty || outputs[0] == null || outputs[1] == null) {
      return [];
    }

    final detsRaw = outputs[0]!.value;
    final labelsRaw = outputs[1]!.value;
    final masksRaw = outputs.length > 2 ? outputs[2]?.value : null;

    // Parse dets: expected shape [1, N, 4] as nested lists
    final List<List<double>> dets = _parseDets(detsRaw);
    final List<double> scores = _parseLabels(labelsRaw);
    final int numDetections = dets.length;

    final List<PigDetection> detections = [];

    // Debug: print first detection's raw values to determine coordinate space
    if (dets.isNotEmpty) {
      print('DEBUG raw dets[0]: ${dets[0]}');
      print('DEBUG originalSize: ${originalWidth}x$originalHeight, inputSize: $inputSize');
    }

    // Determine if coords are normalized [0,1] or in input pixel space [0, inputSize]
    // Check if max values suggest pixel coords (>1) or normalized (<=1)
    bool isNormalized = true;
    for (int i = 0; i < math.min(10, dets.length); i++) {
      if (dets[i].any((v) => v.abs() > 1.5)) {
        isNormalized = false;
        break;
      }
    }

    // Scale factors from input space to original image space
    final double scaleToOrigX = originalWidth / inputSize;
    final double scaleToOrigY = originalHeight / inputSize;

    for (int i = 0; i < numDetections && detections.length < _maxDetections; i++) {
      final score = scores[i];
      if (score < _confidenceThreshold) continue;

      double cx, cy, w, h;
      if (isNormalized) {
        // Normalized [0,1] — scale to original image pixels
        cx = dets[i][0] * originalWidth;
        cy = dets[i][1] * originalHeight;
        w = dets[i][2] * originalWidth;
        h = dets[i][3] * originalHeight;
      } else {
        // Input pixel space [0, inputSize] — scale to original image pixels
        cx = dets[i][0] * scaleToOrigX;
        cy = dets[i][1] * scaleToOrigY;
        w = dets[i][2] * scaleToOrigX;
        h = dets[i][3] * scaleToOrigY;
      }

      final left = (cx - w / 2).clamp(0.0, originalWidth.toDouble());
      final top = (cy - h / 2).clamp(0.0, originalHeight.toDouble());
      final right = (cx + w / 2).clamp(0.0, originalWidth.toDouble());
      final bottom = (cy + h / 2).clamp(0.0, originalHeight.toDouble());

      print('DEBUG det[$i]: score=${score.toStringAsFixed(3)}, box=($left, $top, $right, $bottom)');

      // Parse mask for this detection if available
      List<List<double>>? mask;
      if (masksRaw != null) {
        mask = _parseMask(masksRaw, i, originalWidth, originalHeight);
      }

      detections.add(PigDetection(
        boundingBox: Rect.fromLTRB(left, top, right, bottom),
        confidence: score,
        classId: 0, // Single-class pig detection
        mask: mask,
      ));
    }

    // Sort by confidence descending
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
    if (raw is Float32List || (raw is List && raw.isNotEmpty && raw[0] is num)) {
      final flat = raw is Float32List ? raw : Float32List.fromList((raw as List).cast<double>());
      final n = flat.length ~/ 4;
      return List.generate(n, (i) => [flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2], flat[i * 4 + 3]]);
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

    if (raw is Float32List || (raw is List && raw.isNotEmpty && raw[0] is num)) {
      final flat = raw is Float32List ? raw : Float32List.fromList((raw as List).cast<double>());
      return flat.map((v) => _sigmoid(v)).toList();
    }

    return [];
  }

  /// Parse a single detection's mask from the masks output.
  List<List<double>>? _parseMask(
    dynamic masksRaw,
    int detectionIndex,
    int originalWidth,
    int originalHeight,
  ) {
    try {
      // Expected shape: [1, N, maskH, maskW]
      // Access masksRaw[0][detectionIndex] to get [maskH, maskW]
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
      // Mask parsing failed — detection still valid without mask
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
  }
}
