import 'package:flutter/material.dart';

class ResponsiveUtils {
  static double screenWidth(BuildContext context) {
    return MediaQuery.of(context).size.width;
  }

  static double screenHeight(BuildContext context) {
    return MediaQuery.of(context).size.height;
  }

  // Get responsive width based on percentage of screen width
  static double width(BuildContext context, double percentage) {
    return screenWidth(context) * percentage / 100;
  }

  // Get responsive height based on percentage of screen height
  static double height(BuildContext context, double percentage) {
    return screenHeight(context) * percentage / 100;
  }

  // Get responsive font size based on screen width
  static double fontSize(BuildContext context, double baseSize) {
    double screenWidth = MediaQuery.of(context).size.width;
    double scaleFactor = screenWidth / 375; // Base width (iPhone 6/7/8)
    return baseSize * scaleFactor;
  }

  // Check if device is tablet (width > 600)
  static bool isTablet(BuildContext context) {
    return screenWidth(context) > 600;
  }

  // Get safe area padding
  static EdgeInsets safeAreaPadding(BuildContext context) {
    return MediaQuery.of(context).padding;
  }

  // Get status bar height
  static double statusBarHeight(BuildContext context) {
    return MediaQuery.of(context).padding.top;
  }

  // Get responsive padding
  static EdgeInsets responsivePadding(BuildContext context, {
    double? left,
    double? top,
    double? right,
    double? bottom,
    double? horizontal,
    double? vertical,
    double? all,
  }) {
    double screenWidth = MediaQuery.of(context).size.width;
    double scaleFactor = screenWidth / 375;

    return EdgeInsets.only(
      left: (left ?? horizontal ?? all ?? 0) * scaleFactor,
      top: (top ?? vertical ?? all ?? 0) * scaleFactor,
      right: (right ?? horizontal ?? all ?? 0) * scaleFactor,
      bottom: (bottom ?? vertical ?? all ?? 0) * scaleFactor,
    );
  }
}