import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:blackpig/models/detection_result.dart';
import 'package:blackpig/utils/responsive.dart';

class PigImageWithOverlay extends StatefulWidget {
  final String? imagePath;
  final DetectionResult? detectionResult;

  const PigImageWithOverlay({
    super.key,
    this.imagePath,
    this.detectionResult,
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
                        return Stack(
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
                                ),
                              ),
                          ],
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
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = displayWidth / detectionResult.imageWidth;
    final scaleY = displayHeight / detectionResult.imageHeight;

    for (int i = 0; i < detectionResult.detections.length; i++) {
      final det = detectionResult.detections[i];
      final color = _colors[i % _colors.length];

      // Draw mask overlay if available
      if (det.mask != null) {
        _drawMask(canvas, det, scaleX, scaleY, color);
      }

      // Draw bounding box
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      final scaledBox = Rect.fromLTRB(
        det.boundingBox.left * scaleX,
        det.boundingBox.top * scaleY,
        det.boundingBox.right * scaleX,
        det.boundingBox.bottom * scaleY,
      );
      canvas.drawRect(scaledBox, boxPaint);

      // Draw confidence label
      final label = '${(det.confidence * 100).toStringAsFixed(0)}%';
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

      // Label background
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

    // The mask covers the ENTIRE image (108x108 = full input),
    // so map mask coordinates to full display coordinates
    final cellW = displayWidth / maskW;
    final cellH = displayHeight / maskH;

    final maskPaint = Paint()..color = color.withValues(alpha: 0.3);

    for (int y = 0; y < maskH; y++) {
      for (int x = 0; x < maskW; x++) {
        if (mask[y][x] > 0) {
          canvas.drawRect(
            Rect.fromLTWH(
              x * cellW,
              y * cellH,
              cellW + 0.5, // slight overlap to avoid gaps
              cellH + 0.5,
            ),
            maskPaint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(DetectionOverlayPainter oldDelegate) =>
      oldDelegate.detectionResult != detectionResult;
}
