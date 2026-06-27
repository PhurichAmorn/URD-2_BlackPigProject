import 'package:flutter/material.dart';
import 'package:DooMoo/components/HomePage/Camera.dart';
import 'package:DooMoo/components/HomePage/Upload.dart';
import 'package:DooMoo/utils/config.dart';
import 'package:DooMoo/utils/responsive.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isProcessing = false;

  void _onProcessingChanged(bool processing) {
    setState(() {
      _isProcessing = processing;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: homeAppBar(),
      body: Stack(
        children: [
          SafeArea(
            child: SingleChildScrollView(
              child: Padding(
                padding: ResponsiveUtils.responsivePadding(context, horizontal: 31),
                child: Column(
                  children: [
                    SizedBox(height: ResponsiveUtils.height(context, 7)),
                    Camera(isDisabled: _isProcessing),
                    SizedBox(height: ResponsiveUtils.height(context, 5)),
                    Upload(onProcessingChanged: _onProcessingChanged),
                    SizedBox(height: ResponsiveUtils.height(context, 5)),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            right: 12,
            bottom: 12,
            child: Text(
              'v${AppConfig.version}',
              style: TextStyle(
                fontSize: 20,
                color: Colors.grey[400],
              ),
            ),
          ),
        ],
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
            'DooMoo',
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
