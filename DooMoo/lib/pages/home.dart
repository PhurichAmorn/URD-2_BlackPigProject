import 'package:flutter/material.dart';
import 'package:blackpig/components/HomePage/Camera.dart';
import 'package:blackpig/components/HomePage/Upload.dart';
import 'package:blackpig/utils/responsive.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: homeAppBar(),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: ResponsiveUtils.responsivePadding(context, horizontal: 31),
            child: Column(
              children: [
                SizedBox(height: ResponsiveUtils.height(context, 7)),
                Camera(),
                SizedBox(height: ResponsiveUtils.height(context, 5)),
                Upload(),
                SizedBox(height: ResponsiveUtils.height(context, 5)),
              ],
            ),
          ),
        ),
      ),
    );
  }

  AppBar homeAppBar() {
    return AppBar(
      toolbarHeight: 90,
      backgroundColor: Color(0xFFFFFFFF),
      shadowColor: Colors.black54,
      elevation: 4,
      automaticallyImplyLeading: false,
      title: Builder(
        builder: (context) => Transform.translate(
          offset: const Offset(15.0, 10.0),
          child: Text(
            'Pig Scanner',
            style: TextStyle(
              fontSize: ResponsiveUtils.fontSize(context, 44),
              fontWeight: FontWeight.bold,
              color: Color(0xFF000000),
            ),
          ),
        ),
      ),
    );
  }
}
