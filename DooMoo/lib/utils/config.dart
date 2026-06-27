import 'package:flutter/foundation.dart';
import 'package:DooMoo/utils/flavor.dart';

class AppConfig {
  static bool debugMode = kDebugMode || Flavor.isDev;
  static String version = '1.0.0+1';
}
