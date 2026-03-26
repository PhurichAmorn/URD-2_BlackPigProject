import 'package:DooMoo/components/DetailsPage/PigInfo.dart';
import 'package:DooMoo/components/DetailsPage/CameraMetadataInfo.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:DooMoo/pages/home.dart';
import 'package:DooMoo/components/DetailsPage/PigImageWithOverlay.dart';
import 'package:DooMoo/utils/responsive.dart';
import 'package:DooMoo/utils/camera_metadata.dart';
import 'package:DooMoo/models/detection_result.dart';
import 'package:DooMoo/utils/config.dart';
import 'package:DooMoo/services/pig_detector.dart';

class DetailsPage extends StatefulWidget {
  final String? imagePath;
  final CameraMetadata? cameraMetadata;
  final DetectionResult? detectionResult;

  const DetailsPage({
    super.key,
    this.imagePath,
    this.cameraMetadata,
    this.detectionResult,
  });

  @override
  State<DetailsPage> createState() => _DetailsPageState();
}

class _DetailsPageState extends State<DetailsPage> {
  int? _selectedPigIndex;
  late DetectionResult? _currentDetectionResult;
  bool _isProcessingSegmentation = false;

  @override
  void initState() {
    super.initState();
    _currentDetectionResult = widget.detectionResult;
  }

  Future<void> _onPigSelected(int index) async {
    setState(() {
      _selectedPigIndex = index;
    });

    final det = _currentDetectionResult?.detections[index];
    if (det != null && det.mask == null && widget.imagePath != null) {
      setState(() {
        _isProcessingSegmentation = true;
      });

      // Give UI a frame to show the loading indicator before model loading/inference
      await Future.delayed(const Duration(milliseconds: 50));

      try {
        final rfDetr = await PigDetector.getInstance();

        // Add padding to crop if possible to help RF-DETR
        final padding = det.boundingBox.width * 0.1;
        final imgW = _currentDetectionResult!.imageWidth.toDouble();
        final imgH = _currentDetectionResult!.imageHeight.toDouble();

        final cropRect = Rect.fromLTRB(
          (det.boundingBox.left - padding).clamp(0, imgW),
          (det.boundingBox.top - padding).clamp(0, imgH),
          (det.boundingBox.right + padding).clamp(0, imgW),
          (det.boundingBox.bottom + padding).clamp(0, imgH),
        );

        final rfResult =
            await rfDetr.detect(widget.imagePath!, cropRect: cropRect);

        if (rfResult.detections.isNotEmpty && mounted) {
          final bestSeg = rfResult.detections.first;
          final updatedDetections =
              List<PigDetection>.from(_currentDetectionResult!.detections);
          updatedDetections[index] = det.copyWith(
            mask: bestSeg.mask,
            maskRect: bestSeg.maskRect,
          );

          setState(() {
            _currentDetectionResult = _currentDetectionResult!
                .copyWith(detections: updatedDetections);
          });
        }
      } catch (e) {
        print('Error running segmentation: $e');
      } finally {
        if (mounted) {
          setState(() {
            _isProcessingSegmentation = false;
          });
        }
      }
    }
  }

  void _resetSelection() {
    setState(() {
      _selectedPigIndex = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: _detailsAppBar(context),
      body: SafeArea(
        child: Stack(
          children: [
            SingleChildScrollView(
              child: Column(
                children: [
                  SizedBox(height: ResponsiveUtils.height(context, 3)),
                  PigImageWithOverlay(
                    imagePath: widget.imagePath,
                    detectionResult: _currentDetectionResult,
                    selectedPigIndex: _selectedPigIndex,
                    onPigSelected: _onPigSelected,
                  ),
                  SizedBox(height: ResponsiveUtils.height(context, 4)),
                  PigInfo(
                    detectionResult: _currentDetectionResult,
                    selectedPigIndex: _selectedPigIndex,
                    onReset: _selectedPigIndex != null ? _resetSelection : null,
                    cameraMetadata: widget.cameraMetadata,
                  ),
                  if (AppConfig.debugMode) ...[
                    SizedBox(height: ResponsiveUtils.height(context, 3)),
                    CameraMetadataInfo(cameraMetadata: widget.cameraMetadata),
                  ],
                  SizedBox(height: ResponsiveUtils.height(context, 3)),
                ],
              ),
            ),
            if (_isProcessingSegmentation)
              Positioned.fill(
                child: Container(
                  color: Colors.black26,
                  child: Center(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 32, vertical: 24),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: const [
                          BoxShadow(
                            color: Colors.black26,
                            blurRadius: 10,
                            offset: Offset(0, 4),
                          )
                        ],
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CircularProgressIndicator(
                            color: Color(0xFF2671F4),
                            strokeWidth: 4,
                          ),
                          const SizedBox(height: 20),
                          Text(
                            'กำลังวิเคราะห์ส่วนต่าง ๆ...',
                            style: TextStyle(
                              fontSize: ResponsiveUtils.fontSize(context, 28),
                              fontWeight: FontWeight.bold,
                              color: const Color(0xFF5A5A5A),
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'โปรดรอสักครู่',
                            style: TextStyle(
                              fontSize: ResponsiveUtils.fontSize(context, 22),
                              color: const Color(0xFF999999),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  AppBar _detailsAppBar(BuildContext context) {
    return AppBar(
      toolbarHeight: ResponsiveUtils.height(context, 10),
      backgroundColor: Color(0xFFFFFFFF),
      shadowColor: Colors.black54,
      elevation: 4,
      automaticallyImplyLeading: false,
      title: Padding(
        padding: EdgeInsets.only(right: ResponsiveUtils.width(context, 12)),
        child: Center(
          child: GestureDetector(
            onLongPress: () {
              setState(() {
                AppConfig.debugMode = !AppConfig.debugMode;
              });
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                      'Debug Mode: ${AppConfig.debugMode ? "Enabled" : "Disabled"}'),
                  duration: const Duration(seconds: 1),
                ),
              );
            },
            child: Text(
              'รายละเอียด',
              style: TextStyle(
                fontSize: ResponsiveUtils.fontSize(context, 43),
                fontWeight: FontWeight.bold,
                color: Color(0xFF000000),
              ),
            ),
          ),
        ),
      ),
      leading: GestureDetector(
        onTap: () {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => HomePage()),
          );
        },
        child: Container(
          margin: ResponsiveUtils.responsivePadding(context, left: 25),
          alignment: Alignment.center,
          width: ResponsiveUtils.width(context, 18),
          height: ResponsiveUtils.width(context, 18),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.transparent,
          ),
          child: SvgPicture.asset(
            'assets/icons/ArrowLeftBlack.svg',
            height: ResponsiveUtils.width(context, 13),
            width: ResponsiveUtils.width(context, 13),
          ),
        ),
      ),
    );
  }
}
