import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:image_picker/image_picker.dart';
import 'package:blackpig/pages/details.dart';
import 'package:blackpig/utils/camera_metadata.dart';
import 'package:blackpig/services/pig_detector.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraState();
}

class _CameraState extends State<CameraPage> {
  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _takePhoto();
  }

  Future<void> _takePhoto() async {
    try {
      final XFile? xfile = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 2048,
        imageQuality: 90,
      );
      if (!mounted) return;

      if (xfile == null) {
        // User cancelled - navigate back to HomePage
        Navigator.of(context).pop();
        return;
      }

      setState(() => _isProcessing = true);

      // Extract camera metadata from the captured image
      final metadata =
          await CameraMetadataExtractor.extractFromImage(xfile.path);

      // Run pig detection inference
      final detector = await PigDetector.getInstance();
      final detectionResult = await detector.detect(xfile.path);

      // Debug: Print metadata (you can remove this later)
      print('Camera Metadata: $metadata');
      print('Detection Result: $detectionResult');

      if (!mounted) return;

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
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error opening camera: $e')),
      );
      // Navigate back to HomePage on error as well
      Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.only(left: 31, top: 192),
        child: Container(
          width: 351,
          height: 248,
          decoration: BoxDecoration(
            color: const Color(0xFFFCFCFC),
            borderRadius: BorderRadius.circular(20),
            boxShadow: const [
              BoxShadow(
                color: Color.fromARGB(20, 0, 0, 0),
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
                    Padding(
                      padding: const EdgeInsets.only(top: 30),
                      child: SvgPicture.asset(
                        'assets/icons/Camera.svg',
                        width: 77,
                        height: 77,
                      ),
                    ),
                    if (_isProcessing) ...[
                      const SizedBox(height: 20),
                      const CircularProgressIndicator(),
                      const SizedBox(height: 10),
                      const Text(
                        'กำลังวิเคราะห์...',
                        style: TextStyle(
                          fontSize: 16,
                          color: Color(0xFF5A5A5A),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
