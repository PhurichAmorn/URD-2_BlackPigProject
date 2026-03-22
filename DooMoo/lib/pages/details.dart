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

  void _onPigSelected(int index) {
    setState(() {
      _selectedPigIndex = index;
    });
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
        child: SingleChildScrollView(
          child: Column(
            children: [
              SizedBox(height: ResponsiveUtils.height(context, 3)),
              PigImageWithOverlay(
                imagePath: widget.imagePath,
                detectionResult: widget.detectionResult,
                selectedPigIndex: _selectedPigIndex,
                onPigSelected: _onPigSelected,
              ),
              SizedBox(height: ResponsiveUtils.height(context, 4)),
              PigInfo(
                detectionResult: widget.detectionResult,
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
