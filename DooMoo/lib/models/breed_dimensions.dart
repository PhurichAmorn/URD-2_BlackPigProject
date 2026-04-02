/// Real body length (snout → tail base, top-down view) per breed in mm.
/// Used as a prior for geometric distance estimation.
const Map<String, double> breedRealLengthMm = {
  'black_pig_small': 700.0,
  'black_pig_medium': 900.0,
  'black_pig_large': 1100.0,
  'unknown': 850.0,
};

double breedRealLengthFor(String breedLabel) {
  return breedRealLengthMm[breedLabel] ?? breedRealLengthMm['unknown']!;
}
