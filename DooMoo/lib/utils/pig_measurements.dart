import 'dart:math';
import 'dart:ui';

/// PCA-based pig body measurements in original image pixels.
/// Mirrors the `measure_pig_length_and_widths` function from model.ipynb.
class PigMeasurements {
  final double length;
  final double widthTop; // chest
  final double widthMiddle; // abdominal
  final double widthBottom; // hip

  // Line endpoints in original image coordinates for visualization
  final Offset lengthP1; // main axis start
  final Offset lengthP2; // main axis end
  final Offset topA, topB; // chest width line
  final Offset midA, midB; // abdominal width line
  final Offset botA, botB; // hip width line

  const PigMeasurements({
    required this.length,
    required this.widthTop,
    required this.widthMiddle,
    required this.widthBottom,
    required this.lengthP1,
    required this.lengthP2,
    required this.topA,
    required this.topB,
    required this.midA,
    required this.midB,
    required this.botA,
    required this.botB,
  });

  /// Compute PCA-based measurements from a segmentation mask and its bounding box.
  ///
  /// [mask] is the model output mask `[maskH][maskW]` with values 0.0–1.0.
  /// [boundingBox] is the detection bounding box in original image coordinates.
  /// [fracShift] fraction from each end to measure top/bottom width (default 0.15).
  /// [windowFrac] fraction of length used as local width window (default 0.15).
  static PigMeasurements? fromMask(
    List<List<double>> mask,
    Rect boundingBox, {
    int? imageWidth,
    int? imageHeight,
    Rect? maskRect,
    double fracShift = 0.15,
    double windowFrac = 0.15,
  }) {
    final int maskH = mask.length;
    if (maskH == 0) return null;
    final int maskW = mask[0].length;
    if (maskW == 0) return null;

    final List<double> pointsX = [];
    final List<double> pointsY = [];

    // Use maskRect if provided, otherwise default to full image
    final double imgW = (imageWidth ?? maskW).toDouble();
    final double imgH = (imageHeight ?? maskH).toDouble();
    final Rect activeRect = maskRect ?? Rect.fromLTWH(0, 0, imgW, imgH);

    for (int y = 0; y < maskH; y++) {
      for (int x = 0; x < maskW; x++) {
        if (mask[y][x] <= 0.5) continue;

        // Check if edge pixel
        bool isEdge = (x == 0 ||
            x == maskW - 1 ||
            y == 0 ||
            y == maskH - 1 ||
            mask[y - 1][x] <= 0.5 ||
            mask[y + 1][x] <= 0.5 ||
            mask[y][x - 1] <= 0.5 ||
            mask[y][x + 1] <= 0.5);

        if (isEdge) {
          // Scale to coordinates within the activeRect
          final origX = activeRect.left + (x / maskW) * activeRect.width;
          final origY = activeRect.top + (y / maskH) * activeRect.height;

          // Only include points within the bounding box (tighten the PCA to the specific pig)
          if (origX >= boundingBox.left - 2 &&
              origX <= boundingBox.right + 2 &&
              origY >= boundingBox.top - 2 &&
              origY <= boundingBox.bottom + 2) {
            pointsX.add(origX);
            pointsY.add(origY);
          }
        }
      }
    }

    if (pointsX.length < 2) return null;
    final int n = pointsX.length;

    // 2. PCA — compute mean
    double meanX = 0, meanY = 0;
    for (int i = 0; i < n; i++) {
      meanX += pointsX[i];
      meanY += pointsY[i];
    }
    meanX /= n;
    meanY /= n;

    // Covariance matrix [cxx cxy; cxy cyy]
    double cxx = 0, cxy = 0, cyy = 0;
    for (int i = 0; i < n; i++) {
      final dx = pointsX[i] - meanX;
      final dy = pointsY[i] - meanY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    cxx /= n;
    cxy /= n;
    cyy /= n;

    // Eigendecomposition of 2×2 symmetric matrix
    final trace = cxx + cyy;
    final det = cxx * cyy - cxy * cxy;
    final disc = sqrt(max(0, trace * trace - 4 * det));
    final lambda1 = (trace + disc) / 2; // larger eigenvalue

    // Eigenvector for lambda1 (main axis)
    double ax, ay;
    if (cxy.abs() > 1e-10) {
      ax = lambda1 - cyy;
      ay = cxy;
    } else {
      ax = cxx >= cyy ? 1 : 0;
      ay = cxx >= cyy ? 0 : 1;
    }
    final norm = sqrt(ax * ax + ay * ay);
    if (norm < 1e-10) return null;
    ax /= norm;
    ay /= norm;

    // Perpendicular axis
    final bx = -ay;
    final by = ax;

    // 3. Project all points onto main axis to find endpoints
    double minProj = double.infinity, maxProj = double.negativeInfinity;
    for (int i = 0; i < n; i++) {
      final proj = (pointsX[i] - meanX) * ax + (pointsY[i] - meanY) * ay;
      if (proj < minProj) minProj = proj;
      if (proj > maxProj) maxProj = proj;
    }

    final length = maxProj - minProj;
    if (length <= 0) return null;

    // Main axis endpoints
    final p1 = Offset(meanX + ax * minProj, meanY + ay * minProj);
    final p2 = Offset(meanX + ax * maxProj, meanY + ay * maxProj);

    // 4. Measure widths at top (fracShift), middle (0.5), bottom (1-fracShift)
    final topProj = minProj + length * fracShift;
    final midProj = minProj + length * 0.5;
    final botProj = maxProj - length * fracShift;
    final window = length * windowFrac;

    final topResult = _localWidthWithEndpoints(
        pointsX, pointsY, meanX, meanY, ax, ay, bx, by, topProj, window);
    final midResult = _localWidthWithEndpoints(
        pointsX, pointsY, meanX, meanY, ax, ay, bx, by, midProj, window);
    final botResult = _localWidthWithEndpoints(
        pointsX, pointsY, meanX, meanY, ax, ay, bx, by, botProj, window);

    // Sample points along main axis for width line centers
    final topCenter = Offset(meanX + ax * topProj, meanY + ay * topProj);
    final midCenter = Offset(meanX + ax * midProj, meanY + ay * midProj);
    final botCenter = Offset(meanX + ax * botProj, meanY + ay * botProj);

    return PigMeasurements(
      length: length,
      widthTop: topResult.width,
      widthMiddle: midResult.width,
      widthBottom: botResult.width,
      lengthP1: p1,
      lengthP2: p2,
      topA: Offset(topCenter.dx + bx * topResult.minPerp,
          topCenter.dy + by * topResult.minPerp),
      topB: Offset(topCenter.dx + bx * topResult.maxPerp,
          topCenter.dy + by * topResult.maxPerp),
      midA: Offset(midCenter.dx + bx * midResult.minPerp,
          midCenter.dy + by * midResult.minPerp),
      midB: Offset(midCenter.dx + bx * midResult.maxPerp,
          midCenter.dy + by * midResult.maxPerp),
      botA: Offset(botCenter.dx + bx * botResult.minPerp,
          botCenter.dy + by * botResult.minPerp),
      botB: Offset(botCenter.dx + bx * botResult.maxPerp,
          botCenter.dy + by * botResult.maxPerp),
    );
  }

  /// Compute the width perpendicular to the main axis at a given projection,
  /// returning both the width and the min/max perpendicular projections.
  static _WidthResult _localWidthWithEndpoints(
    List<double> pointsX,
    List<double> pointsY,
    double meanX,
    double meanY,
    double ax,
    double ay,
    double bx,
    double by,
    double targetProj,
    double window,
  ) {
    double minPerp = double.infinity;
    double maxPerp = double.negativeInfinity;
    bool found = false;

    for (int i = 0; i < pointsX.length; i++) {
      final dx = pointsX[i] - meanX;
      final dy = pointsY[i] - meanY;
      final proj = dx * ax + dy * ay;
      if ((proj - targetProj).abs() <= window) {
        final perp = dx * bx + dy * by;
        if (perp < minPerp) minPerp = perp;
        if (perp > maxPerp) maxPerp = perp;
        found = true;
      }
    }

    if (!found) {
      return _WidthResult(width: 0, minPerp: 0, maxPerp: 0);
    }
    return _WidthResult(
      width: maxPerp - minPerp,
      minPerp: minPerp,
      maxPerp: maxPerp,
    );
  }
}

class _WidthResult {
  final double width;
  final double minPerp;
  final double maxPerp;
  const _WidthResult({
    required this.width,
    required this.minPerp,
    required this.maxPerp,
  });
}
