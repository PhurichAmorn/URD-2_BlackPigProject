import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:image_picker/image_picker.dart';
import 'package:DooMoo/pages/details.dart';
import 'package:DooMoo/utils/responsive.dart';
import 'package:DooMoo/utils/camera_metadata.dart';
import 'package:DooMoo/services/yolo_detector.dart';
import 'package:DooMoo/models/detection_result.dart';

class Upload extends StatefulWidget {
  final ValueChanged<bool>? onProcessingChanged;
  const Upload({super.key, this.onProcessingChanged});

  @override
  State<Upload> createState() => _UploadState();
}

class _UploadState extends State<Upload> {
  bool _isProcessing = false;

  void _setProcessing(bool processing) {
    if (mounted) {
      setState(() {
        _isProcessing = processing;
      });
      widget.onProcessingChanged?.call(processing);
    }
  }

  Future<void> _handleUpload() async {
    final picker = ImagePicker();
    final XFile? xfile = await picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 2048,
      imageQuality: 90,
    );
    if (xfile != null && mounted) {
      _setProcessing(true);

      try {
        // Extract camera metadata from the uploaded image
        final metadata =
            await CameraMetadataExtractor.extractFromImage(xfile.path);

        // Run YOLO pig detection inference
        final detector = await YoloDetector.getInstance();
        final detections = await detector.detect(xfile.path);

        final detectionResult = DetectionResult(
          detections: detections,
          imageWidth: metadata.imageWidth ?? 0,
          imageHeight: metadata.imageHeight ?? 0,
        );

        // Debug: Print metadata (you can remove this later)
        print('Camera Metadata: $metadata');
        print('Detection Result: $detectionResult');

        if (!mounted) return;
        _setProcessing(false);

        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => DetailsPage(
              imagePath: xfile.path,
              cameraMetadata: metadata,
              detectionResult: detectionResult,
            ),
          ),
        );
      } catch (e) {
        print('Upload/Detection Error: $e');
        if (mounted) {
          _setProcessing(false);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('เกิดข้อผิดพลาด: $e')),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: ResponsiveUtils.width(context, 90),
      height: ResponsiveUtils.height(context, 30),
      decoration: BoxDecoration(
        color: const Color(0xFFFCFCFC),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Color.fromARGB(25, 0, 0, 0),
            blurRadius: 4,
            spreadRadius: 1,
            offset: Offset(0, 2),
          )
        ],
      ),
      child: Stack(
        children: [
          Align(
            alignment: Alignment.topCenter,
            child: Column(
              children: [
                uploadLogo(context),
                _isProcessing ? processingText(context) : text(context),
                _isProcessing ? loadingIndicator(context) : button(context),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget loadingIndicator(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: Column(
        children: [
          SizedBox(
            width: ResponsiveUtils.width(context, 8),
            height: ResponsiveUtils.width(context, 8),
            child: const CircularProgressIndicator(
              strokeWidth: 3,
              color: Color(0xFF2671F4),
            ),
          ),
        ],
      ),
    );
  }

  Widget processingText(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(top: 3),
      child: Text(
        'กำลังวิเคราะห์...',
        style: TextStyle(
          fontSize: ResponsiveUtils.fontSize(context, 34),
          fontWeight: FontWeight.bold,
          color: Color(0xFF2671F4),
        ),
      ),
    );
  }

  Padding button(BuildContext context) {
    return Padding(
        padding: const EdgeInsets.only(top: 8),
        child: GestureDetector(
          onTap: _isProcessing ? null : _handleUpload,
          child: Container(
            width: ResponsiveUtils.width(context, 60),
            height: ResponsiveUtils.height(context, 7),
            decoration: BoxDecoration(
                color: Color(0xFFFFFFFF),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                    color: _isProcessing ? Colors.grey : Color(0xFF2671F4),
                    width: 2),
                boxShadow: [
                  BoxShadow(
                    color: Color.fromARGB(25, 0, 0, 0),
                    blurRadius: 4,
                    offset: Offset(0, 4),
                  )
                ]),
            child: Center(
              child: Text(
                'คลิก',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 24),
                  fontWeight: FontWeight.bold,
                  color: _isProcessing ? Colors.grey : Color(0xFF2671F4),
                ),
              ),
            ),
          ),
        ));
  }

  Padding text(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(top: 3),
      child: Text(
        'อัพโหลดรูปหมู',
        style: TextStyle(
          fontSize: ResponsiveUtils.fontSize(context, 34),
          fontWeight: FontWeight.bold,
          color: Color(0xFF5A5A5A),
        ),
      ),
    );
  }

  Padding uploadLogo(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 30),
      child: SvgPicture.asset(
        'assets/icons/Upload.svg',
        width: ResponsiveUtils.width(context, 20),
        height: ResponsiveUtils.width(context, 20),
      ),
    );
  }
}
