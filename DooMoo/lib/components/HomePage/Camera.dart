import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:DooMoo/pages/camera.dart';
import 'package:DooMoo/utils/responsive.dart';

class Camera extends StatelessWidget {
  final bool isDisabled;
  const Camera({super.key, this.isDisabled = false});

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
            offset: Offset(0, 2), // Shadow position
          )
        ],
      ),
      child: Stack(
        children: [
          Align(
            alignment: Alignment.topCenter,
            child: Column(
              children: [
                cameraLogo(context),
                text(context),
                button(context),
              ],
            ),
          ),
          if (isDisabled)
            Container(
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(20),
              ),
            ),
        ],
      ),
    );
  }

  Padding button(BuildContext context) {
    return Padding(
        padding: const EdgeInsets.only(top: 8),
        child: GestureDetector(
          onTap: isDisabled
              ? null
              : () {
                  Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const CameraPage()));
                },
          child: Container(
            width: ResponsiveUtils.width(context, 60),
            height: ResponsiveUtils.height(context, 7),
            decoration: BoxDecoration(
                color: isDisabled ? Colors.grey : Color(0xFF2671F4),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Color.fromARGB(25, 0, 0, 0),
                    blurRadius: 4,
                    offset: Offset(0, 4), // Shadow position
                  )
                ]),
            child: Center(
              child: Text(
                'คลิก',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 24),
                  fontWeight: FontWeight.bold,
                  color: Color(0xFFFFFFFF),
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
        'ถ่ายรูปหมู',
        style: TextStyle(
          fontSize: ResponsiveUtils.fontSize(context, 34),
          fontWeight: FontWeight.bold,
          color: isDisabled ? Colors.grey : Color(0xFF5A5A5A),
        ),
      ),
    );
  }

  Padding cameraLogo(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 30),
      child: Opacity(
        opacity: isDisabled ? 0.5 : 1.0,
        child: SvgPicture.asset(
          'assets/icons/Camera.svg',
          width: ResponsiveUtils.width(context, 20),
          height: ResponsiveUtils.width(context, 20),
        ),
      ),
    );
  }
}
