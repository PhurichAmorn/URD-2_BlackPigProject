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

      test('when pixel length changes then result scales proportionally', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 3000;
    final imageHeight = 4000;
    final distanceMm = 1000.0;
    final pixelLength = 500.0;

    final resultMm = PigMath.pixelToMm(
      pixelLength: pixelLength,
      distanceMm: distanceMm,
      focalLength: focalLength,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );

    expect(resultMm, closeTo(190.8395, 0.001));
  });

    test('when pixel length is zero then returns zero', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 3000;
    final imageHeight = 4000;
    final distanceMm = 1000.0;
    final pixelLength = 0.0;

    final resultMm = PigMath.pixelToMm(
      pixelLength: pixelLength,
      distanceMm: distanceMm,
      focalLength: focalLength,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );

    expect(resultMm, 0);
  });

    test('when distance increases then real world size increases', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 3000;
    final imageHeight = 4000;
    final distanceMm = 2000.0;
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

    expect(resultMm, greaterThan(381.679));
  });

  test('when pixel length is very small then still computes value', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 3000;
    final imageHeight = 4000;
    final distanceMm = 1000.0;
    final pixelLength = 0.0001;

    final resultMm = PigMath.pixelToMm(
      pixelLength: pixelLength,
      distanceMm: distanceMm,
      focalLength: focalLength,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );

    expect(resultMm, greaterThan(0));
  });

  test('when pixel length is extremely large then still returns value', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 3000;
    final imageHeight = 4000;
    final distanceMm = 1000.0;
    final pixelLength = 1000000.0;

    final resultMm = PigMath.pixelToMm(
      pixelLength: pixelLength,
      distanceMm: distanceMm,
      focalLength: focalLength,
      sensorWidth: sensorWidth,
      sensorHeight: sensorHeight,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );

    expect(resultMm, isNotNull);
  });

    test('when focal length is zero then returns null', () {
    final focalLength = 0.0;
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

    expect(resultMm, isNull);
  });

  test('when image width is zero then returns null', () {
    final focalLength = 5.24;
    final sensorWidth = 7.68;
    final sensorHeight = 5.76;
    final imageWidth = 0;
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

    expect(resultMm, isNull);
  });

  test('when sensor width is zero then returns null', () {
    final focalLength = 5.24;
    final sensorWidth = 0.0;
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

    expect(resultMm, isNull);
  });

    test('when distance is zero then returns null', () {
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
