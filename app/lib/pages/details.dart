import 'package:blackpig/components/DetailsPage/PigInfo.dart';
import 'package:blackpig/components/DetailsPage/CameraMetadataInfo.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:blackpig/pages/home.dart';
import 'package:blackpig/components/DetailsPage/PigImageWithOverlay.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/utils/camera_metadata.dart';
import 'package:blackpig/models/detection_result.dart';

class DetailsPage extends StatelessWidget {
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
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: detailsAppBar(context),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              SizedBox(height: ResponsiveUtils.height(context, 3)),
              PigImageWithOverlay(
                imagePath: imagePath,
                detectionResult: detectionResult,
              ),
              SizedBox(height: ResponsiveUtils.height(context, 4)),
              PigInfo(detectionResult: detectionResult),
              SizedBox(height: ResponsiveUtils.height(context, 3)),
              CameraMetadataInfo(cameraMetadata: cameraMetadata),
              SizedBox(height: ResponsiveUtils.height(context, 3)),
            ],
          ),
        ),
      ),
    );
  }

  AppBar detailsAppBar(BuildContext context) {
    return AppBar(
      toolbarHeight: ResponsiveUtils.height(context, 10),
      backgroundColor: Color(0xFFFFFFFF),
      shadowColor: Colors.black54,
      elevation: 4,
      automaticallyImplyLeading: false,
      title: Padding(
        padding: EdgeInsets.only(right: ResponsiveUtils.width(context, 12)),
        child: Center(
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
