class PigMath {
  /// Estimate weight using regression model:
  /// Weight = -21.95431 + 0.31079(Body Length) + 0.43166(Chest Width)
  ///        + 0.47990(Abdominal Width) + 0.42656(Hip Width)
  /// All inputs in mm (converted to cm internally), output in kg.
  static String estimateWeight({
    required double? bodyLengthMm,
    required double? chestWidthMm,
    required double? abdominalWidthMm,
    required double? hipWidthMm,
  }) {
    if (bodyLengthMm == null ||
        chestWidthMm == null ||
        abdominalWidthMm == null ||
        hipWidthMm == null) return '-';
    final weight = -21.95431
        + 0.31079 * (bodyLengthMm / 10)
        + 0.43166 * (chestWidthMm / 10)
        + 0.47990 * (abdominalWidthMm / 10)
        + 0.42656 * (hipWidthMm / 10);
    if (weight < 0) return '-';
    return '${weight.toStringAsFixed(1)} kg';
  }
}
