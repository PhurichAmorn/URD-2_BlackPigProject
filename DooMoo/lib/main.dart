import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:DooMoo/pages/home.dart';
import 'package:DooMoo/utils/camera_metadata.dart';
import 'package:DooMoo/services/pig_detector.dart';
import 'package:DooMoo/services/yolo_detector.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock orientation to portrait only
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Initialize camera hardware metadata cache on first launch
  await CameraMetadataCache.initializeHardwareMetadata();

  // Pre-load ONNX models in background (fire-and-forget)
  YoloDetector.getInstance();
  PigDetector.getInstance();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(fontFamily: 'DB HelvethaicaMon X'),
      home: HomePage(),
    );
  }
}
