import 'package:flutter_test/flutter_test.dart';
import 'package:DooMoo/utils/pig_math.dart';

void main() {
  group('PigMath.estimateWeight', () {
    test('test_regression_model_when_input_is_null_then_output_is_dash', () {
      final bodyLength = 100.0;
      final chestWidth = 100.0;
      final abdominalWidth = 100.0;
      final hipWidth = 100.0;

      final result1 = PigMath.estimateWeight(bodyLengthMm: null, chestWidthMm: chestWidth, abdominalWidthMm: abdominalWidth, hipWidthMm: hipWidth);
      final result2 = PigMath.estimateWeight(bodyLengthMm: bodyLength, chestWidthMm: null, abdominalWidthMm: abdominalWidth, hipWidthMm: hipWidth);
      final result3 = PigMath.estimateWeight(bodyLengthMm: bodyLength, chestWidthMm: chestWidth, abdominalWidthMm: null, hipWidthMm: hipWidth);
      final result4 = PigMath.estimateWeight(bodyLengthMm: bodyLength, chestWidthMm: chestWidth, abdominalWidthMm: abdominalWidth, hipWidthMm: null);

      expect(result1, '-');
      expect(result2, '-');
      expect(result3, '-');
      expect(result4, '-');
    });

    test('test_regression_model_when_input_is_zero_then_output_is_dash', () {
      final zero = 0.0;

      final result = PigMath.estimateWeight(bodyLengthMm: zero, chestWidthMm: zero, abdominalWidthMm: zero, hipWidthMm: zero);

      expect(result, '-');
    });

    test('test_regression_model_calculates_correct_weight_for_valid_inputs', () {
      final bodyLength = 450.0;
      final chestWidth = 150.0;
      final abdominalWidth = 180.0;
      final hipWidth = 160.0;

      final result = PigMath.estimateWeight(bodyLengthMm: bodyLength, chestWidthMm: chestWidth, abdominalWidthMm: abdominalWidth, hipWidthMm: hipWidth);

      // Weight = -21.95431 + 0.31079*45 + 0.43166*15 + 0.47990*18 + 0.42656*16 = 13.9693 kg
      expect(result, '14.0 kg');
    });
  });
}
