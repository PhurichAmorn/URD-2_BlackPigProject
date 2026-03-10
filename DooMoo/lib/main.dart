import 'package:flutter/material.dart';
import 'package:blackpig/pages/home.dart';
import 'package:blackpig/utils/camera_metadata.dart';
import 'package:blackpig/services/pig_detector.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize camera hardware metadata cache on first launch
  await CameraMetadataCache.initializeHardwareMetadata();

  // Pre-load ONNX model in background (fire-and-forget)
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
