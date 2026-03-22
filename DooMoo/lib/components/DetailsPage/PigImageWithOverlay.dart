import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:DooMoo/models/detection_result.dart';
import 'package:DooMoo/utils/responsive.dart';
import 'package:DooMoo/utils/pig_measurements.dart';
import 'package:DooMoo/utils/config.dart';

class PigImageWithOverlay extends StatefulWidget {
  final String? imagePath;
  final DetectionResult? detectionResult;
  final int? selectedPigIndex;
  final ValueChanged<int>? onPigSelected;

  const PigImageWithOverlay({
    super.key,
    this.imagePath,
    this.detectionResult,
    this.selectedPigIndex,
    this.onPigSelected,
  });

  @override
  State<PigImageWithOverlay> createState() => _PigImageWithOverlayState();
}

class _PigImageWithOverlayState extends State<PigImageWithOverlay> {
  double? _aspectRatio;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    if (widget.imagePath != null) {
      _loadImageAspectRatio();
    } else {
      _isLoading = false;
    }
  }

  @override
  void didUpdateWidget(PigImageWithOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.imagePath != oldWidget.imagePath) {
      if (widget.imagePath != null) {
        _loadImageAspectRatio();
      } else {
        setState(() {
          _aspectRatio = null;
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _loadImageAspectRatio() async {
    if (widget.imagePath == null) return;

    setState(() => _isLoading = true);

    try {
      final file = File(widget.imagePath!);
      if (!await file.exists()) {
        setState(() {
          _aspectRatio = 16 / 9;
          _isLoading = false;
        });
        return;
      }

      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      if (mounted) {
        setState(() {
          _aspectRatio = frame.image.width / frame.image.height;
          _isLoading = false;
        });
        frame.image.dispose();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _aspectRatio = 16 / 9;
          _isLoading = false;
        });
      }
    }
  }

  void _onTapUp(TapUpDetails details, BoxConstraints constraints) {
    final result = widget.detectionResult;
    if (result == null || result.isEmpty || widget.selectedPigIndex != null) {
      return;
    }

    final tapPos = details.localPosition;
    final displayWidth = constraints.maxWidth;
    final displayHeight = constraints.maxWidth / _aspectRatio!;
    final scaleX = displayWidth / result.imageWidth;
    final scaleY = displayHeight / result.imageHeight;

    for (int i = 0; i < result.detections.length; i++) {
      final det = result.detections[i];
      final scaledBox = Rect.fromLTRB(
        det.boundingBox.left * scaleX,
        det.boundingBox.top * scaleY,
        det.boundingBox.right * scaleX,
        det.boundingBox.bottom * scaleY,
      );
      if (scaledBox.contains(tapPos)) {
        widget.onPigSelected?.call(i);
        return;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: ResponsiveUtils.responsivePadding(context, horizontal: 28),
      child: _isLoading
          ? Container(
              width: ResponsiveUtils.width(context, 90),
              height: ResponsiveUtils.height(context, 32),
              decoration: BoxDecoration(
                color: const Color(0xFFD9D9D9),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(child: CircularProgressIndicator()),
            )
          : _aspectRatio != null
              ? ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: AspectRatio(
                    aspectRatio: _aspectRatio!,
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        return GestureDetector(
                          onTapUp: (details) =>
                              _onTapUp(details, constraints),
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              // Base image
                              Image.file(
                                File(widget.imagePath!),
                                fit: BoxFit.cover,
                                width: constraints.maxWidth,
                              ),
                              // Detection overlay
                              if (widget.detectionResult != null &&
                                  !widget.detectionResult!.isEmpty)
                                CustomPaint(
                                  size: Size(
                                    constraints.maxWidth,
                                    constraints.maxWidth / _aspectRatio!,
                                  ),
                                  painter: DetectionOverlayPainter(
                                    detectionResult: widget.detectionResult!,
                                    displayWidth: constraints.maxWidth,
                                    displayHeight:
                                        constraints.maxWidth / _aspectRatio!,
                                    selectedIndex: widget.selectedPigIndex,
                                  ),
                                ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                )
              : Container(
                  width: ResponsiveUtils.width(context, 90),
                  height: ResponsiveUtils.height(context, 32),
                  decoration: BoxDecoration(
                    color: const Color(0xFFD9D9D9),
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
    );
  }
}

class DetectionOverlayPainter extends CustomPainter {
  final DetectionResult detectionResult;
  final double displayWidth;
  final double displayHeight;
  final int? selectedIndex;

  static const List<Color> _colors = [
    Color(0xFF2671F4),
    Color(0xFFFF6B35),
    Color(0xFF00C853),
    Color(0xFFAA00FF),
    Color(0xFFFFD600),
  ];

  DetectionOverlayPainter({
    required this.detectionResult,
    required this.displayWidth,
    required this.displayHeight,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = displayWidth / detectionResult.imageWidth;
    final scaleY = displayHeight / detectionResult.imageHeight;

    if (selectedIndex == null) {
      // Step 1: Draw all bboxes only (no masks)
      for (int i = 0; i < detectionResult.detections.length; i++) {
        final det = detectionResult.detections[i];
        final color = _colors[i % _colors.length];
        _drawBbox(canvas, det, i, scaleX, scaleY, color, false);
      }
    } else {
      // Step 2: Draw only selected pig's bbox (highlighted) + mask + PCA lines
      final i = selectedIndex!;
      if (i < detectionResult.detections.length) {
        final det = detectionResult.detections[i];
        final color = _colors[i % _colors.length];

        if (det.mask != null) {
          _drawMask(canvas, det, scaleX, scaleY, color);
          
          if (AppConfig.debugMode) {
            // Draw PCA measurement lines only in debug mode
            final pca = PigMeasurements.fromMask(
              det.mask!, det.boundingBox,
              imageWidth: detectionResult.imageWidth,
              imageHeight: detectionResult.imageHeight,
            );
            if (pca != null) {
              _drawPcaLines(canvas, pca, scaleX, scaleY);
            }
          }
        }
        _drawBbox(canvas, det, i, scaleX, scaleY, color, true);
      }
    }
  }

  void _drawBbox(
    Canvas canvas,
    PigDetection det,
    int index,
    double scaleX,
    double scaleY,
    Color color,
    bool highlighted,
  ) {
    final boxPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = highlighted ? 4.0 : 2.5;

    final scaledBox = Rect.fromLTRB(
      det.boundingBox.left * scaleX,
      det.boundingBox.top * scaleY,
      det.boundingBox.right * scaleX,
      det.boundingBox.bottom * scaleY,
    );
    canvas.drawRect(scaledBox, boxPaint);

    // Draw label
    String label;
    if (AppConfig.debugMode) {
      label = highlighted
          ? 'หมู #${index + 1} · ${(det.confidence * 100).toStringAsFixed(0)}%'
          : '${(det.confidence * 100).toStringAsFixed(0)}%';
    } else {
      label = highlighted ? 'หมู #${index + 1}' : '';
    }

    if (label.isNotEmpty) {
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final bgRect = Rect.fromLTWH(
        scaledBox.left,
        scaledBox.top - textPainter.height - 4,
        textPainter.width + 8,
        textPainter.height + 4,
      );
      canvas.drawRect(bgRect, Paint()..color = color);
      textPainter.paint(canvas, Offset(bgRect.left + 4, bgRect.top + 2));
    }
  }

  void _drawMask(
    Canvas canvas,
    PigDetection det,
    double scaleX,
    double scaleY,
    Color color,
  ) {
    final mask = det.mask!;
    if (mask.isEmpty) return;

    final maskH = mask.length;
    final maskW = mask[0].length;

    final cellW = displayWidth / maskW;
    final cellH = displayHeight / maskH;

    // Clip mask rendering to the bounding box
    final scaledBox = Rect.fromLTRB(
      det.boundingBox.left * scaleX,
      det.boundingBox.top * scaleY,
      det.boundingBox.right * scaleX,
      det.boundingBox.bottom * scaleY,
    );
    canvas.save();
    canvas.clipRect(scaledBox);

    final maskPaint = Paint()..color = color.withValues(alpha: 0.3);

    for (int y = 0; y < maskH; y++) {
      for (int x = 0; x < maskW; x++) {
        if (mask[y][x] > 0) {
          canvas.drawRect(
            Rect.fromLTWH(
              x * cellW,
              y * cellH,
              cellW + 0.5,
              cellH + 0.5,
            ),
            maskPaint,
          );
        }
      }
    }

    canvas.restore();
  }

  void _drawPcaLines(
    Canvas canvas,
    PigMeasurements pca,
    double scaleX,
    double scaleY,
  ) {
    Offset scale(Offset p) => Offset(p.dx * scaleX, p.dy * scaleY);

    // Main axis (body length) — red
    final lengthPaint = Paint()
      ..color = const Color(0xFFFF0000)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawLine(scale(pca.lengthP1), scale(pca.lengthP2), lengthPaint);

    // Chest width (top) — blue
    final chestPaint = Paint()
      ..color = const Color(0xFF0000FF)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawLine(scale(pca.topA), scale(pca.topB), chestPaint);

    // Abdominal width (middle) — yellow
    final abdominalPaint = Paint()
      ..color = const Color(0xFFFFFF00)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawLine(scale(pca.midA), scale(pca.midB), abdominalPaint);

    // Hip width (bottom) — orange
    final hipPaint = Paint()
      ..color = const Color(0xFFFF8800)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    canvas.drawLine(scale(pca.botA), scale(pca.botB), hipPaint);

    // Draw labels at endpoints to avoid overlap
    _drawLineLabel(canvas, scale(pca.lengthP2), 'Length',
        const Color(0xFFFF0000), const Offset(5, -5));
    _drawLineLabel(canvas, scale(pca.topA), 'Chest',
        const Color(0xFF0000FF), const Offset(-50, -5));
    _drawLineLabel(canvas, scale(pca.midA), 'Abdominal',
        const Color(0xFFFFFF00), const Offset(-65, -5));
    _drawLineLabel(canvas, scale(pca.botA), 'Hip',
        const Color(0xFFFF8800), const Offset(-35, -5));
  }

  void _drawLineLabel(
    Canvas canvas,
    Offset anchor,
    String label,
    Color color,
    Offset offset,
  ) {
    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final pos = anchor + offset;
    final bgRect = Rect.fromLTWH(
      pos.dx - 3,
      pos.dy - 2,
      tp.width + 6,
      tp.height + 4,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(3)),
      Paint()..color = color.withValues(alpha: 0.8),
    );
    tp.paint(canvas, Offset(bgRect.left + 3, bgRect.top + 2));
  }

  @override
  bool shouldRepaint(DetectionOverlayPainter oldDelegate) =>
      oldDelegate.detectionResult != detectionResult ||
      oldDelegate.selectedIndex != selectedIndex;
}
