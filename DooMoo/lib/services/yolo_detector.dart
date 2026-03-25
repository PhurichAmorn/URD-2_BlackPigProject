import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:image/image.dart' as img;
import 'package:DooMoo/models/detection_result.dart';
import 'dart:ui';

class YoloDetector {
  static YoloDetector? _instance;
  static Future<YoloDetector>? _initFuture;
  static bool _envInitialized = false;
  OrtSession? _session;
  bool _isReady = false;

  static const String _modelAsset = 'assets/models/yolov8_pig.onnx';
  static const int _inputSize = 640;
  static const double _confidenceThreshold = 0.25;
  static const double _iouThreshold = 0.45;

  YoloDetector._();

  static Future<YoloDetector> getInstance() async {
    if (_instance != null && _instance!._isReady) return _instance!;
    
    if (_initFuture != null) return _initFuture!;

    _initFuture = _initInstance();
    return _initFuture!;
  }

  static Future<YoloDetector> _initInstance() async {
    try {
      final detector = YoloDetector._();
      await detector._loadModel();
      _instance = detector;
      return detector;
    } catch (e) {
      _initFuture = null; // Reset future so we can retry
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

  Future<List<PigDetection>> detect(String imagePath) async {
    if (!_isReady || _session == null) {
      // Try one last time to load if not ready
      await _loadModel();
      if (!_isReady || _session == null) {
        throw StateError('YoloDetector not ready after manual load attempt');
      }
    }

    final bytes = await File(imagePath).readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) return [];

    final originalWidth = image.width;
    final originalHeight = image.height;

    // Preprocess: Resize to 640x640 and normalize (NCHW)
    final resized = img.copyResize(image, width: _inputSize, height: _inputSize);
    
    final channelSize = _inputSize * _inputSize;
    final inputData = Float32List(1 * 3 * channelSize);
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        final idx = y * _inputSize + x;
        inputData[0 * channelSize + idx] = pixel.r / 255.0;
        inputData[1 * channelSize + idx] = pixel.g / 255.0;
        inputData[2 * channelSize + idx] = pixel.b / 255.0;
      }
    }

    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, _inputSize, _inputSize],
    );

    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(
      runOptions,
      {'images': inputTensor}, // YOLOv8 input name is typically 'images'
    );

    if (outputs == null || outputs.isEmpty || outputs[0] == null) {
      inputTensor.release();
      runOptions.release();
      return [];
    }

    // YOLOv8 output is typically [1, 4 + num_classes, 8400]
    final outputValue = outputs[0]!.value as List<List<List<double>>>;
    final result = _postprocess(outputValue[0], originalWidth, originalHeight);

    // Cleanup
    inputTensor.release();
    runOptions.release();
    for (var out in outputs) {
      out?.release();
    }

    return result;
  }

  List<PigDetection> _postprocess(List<List<double>> output, int imgW, int imgH) {
    // output is [5, 8400]
    // rows: 0:cx, 1:cy, 2:w, 3:h, 4:score (for 1 class)
    List<PigDetection> candidates = [];
    final int numBoxes = output[0].length;

    for (int i = 0; i < numBoxes; i++) {
      final score = output[4][i];
      if (score < _confidenceThreshold) continue;

      final cx = output[0][i] * imgW / _inputSize;
      final cy = output[1][i] * imgH / _inputSize;
      final w = output[2][i] * imgW / _inputSize;
      final h = output[3][i] * imgH / _inputSize;

      final left = (cx - w / 2).clamp(0.0, imgW.toDouble());
      final top = (cy - h / 2).clamp(0.0, imgH.toDouble());
      final right = (cx + w / 2).clamp(0.0, imgW.toDouble());
      final bottom = (cy + h / 2).clamp(0.0, imgH.toDouble());

      candidates.add(PigDetection(
        boundingBox: Rect.fromLTRB(left, top, right, bottom),
        confidence: score,
        classId: 0,
      ));
    }

    return _nms(candidates);
  }

  List<PigDetection> _nms(List<PigDetection> boxes) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<PigDetection> selected = [];
    List<bool> active = List.filled(boxes.length, true);

    for (int i = 0; i < boxes.length; i++) {
      if (!active[i]) continue;
      selected.add(boxes[i]);
      for (int j = i + 1; j < boxes.length; j++) {
        if (!active[j]) continue;
        if (_iou(boxes[i].boundingBox, boxes[j].boundingBox) > _iouThreshold) {
          active[j] = false;
        }
      }
    }
    return selected;
  }

  double _iou(Rect a, Rect b) {
    final intersection = a.intersect(b);
    if (intersection.width <= 0 || intersection.height <= 0) return 0.0;
    final intersectionArea = intersection.width * intersection.height;
    final unionArea = a.width * a.height + b.width * b.height - intersectionArea;
    return intersectionArea / unionArea;
  }

  void dispose() {
    _session?.release();
    _session = null;
    _isReady = false;
    _instance = null;
  }
}
