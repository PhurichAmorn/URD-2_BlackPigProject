import 'package:flutter_test/flutter_test.dart';
import 'package:DooMoo/utils/pig_math.dart';

void main() {
  group('test pixel to metric conversion', () {
    test('when inputs are valid then calculates correct metric value', () {
      // Example Xiaomi Note 14 metadata
      final focalLength = 5.24;
      final sensorWidth = 7.68;
      final sensorHeight = 5.76;
      final imageWidth = 3000;
      final imageHeight = 4000;
      final distanceMm = 1000.0;
      final pixelLength = 1000.0;

      final resultMm = PigMath.pixelToMm(
        pixelLength: pixelLength,
        distanceMm: distanceMm,
        focalLength: focalLength,
        sensorWidth: sensorWidth,
        sensorHeight: sensorHeight,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      );

      expect(resultMm, closeTo(381.679, 0.001));
    });

    test('when distance is zero or negative then returns null', () {
      final focalLength = 5.24;
      final sensorWidth = 7.68;
      final sensorHeight = 5.76;
      final imageWidth = 3000;
      final imageHeight = 4000;
      final pixelLength = 1000.0;
      final distanceMm = 0.0;

      final resultZero = PigMath.pixelToMm(
        pixelLength: pixelLength,
        distanceMm: distanceMm,
        focalLength: focalLength,
        sensorWidth: sensorWidth,
        sensorHeight: sensorHeight,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      );

      expect(resultZero, isNull);
    });
    
    test('when distance is negative then returns null', () {
      final focalLength = 5.24;
      final sensorWidth = 7.68;
      final sensorHeight = 5.76;
      final imageWidth = 3000;
      final imageHeight = 4000;
      final pixelLength = 1000.0;
      final distanceMm = -1.0;

      final resultNegative = PigMath.pixelToMm(
        pixelLength: pixelLength,
        distanceMm: distanceMm,
        focalLength: focalLength,
        sensorWidth: sensorWidth,
        sensorHeight: sensorHeight,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      );

      expect(resultNegative, isNull);
    });
    });
    }