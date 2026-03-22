import 'package:flutter/material.dart';
import 'package:DooMoo/utils/responsive.dart';
import 'package:DooMoo/models/detection_result.dart';
import 'package:DooMoo/utils/camera_metadata.dart';
import 'package:DooMoo/utils/pig_measurements.dart';
import 'package:DooMoo/utils/config.dart';

class PigInfo extends StatefulWidget {
  final DetectionResult? detectionResult;
  final int? selectedPigIndex;
  final VoidCallback? onReset;
  final CameraMetadata? cameraMetadata;

  const PigInfo({
    super.key,
    this.detectionResult,
    this.selectedPigIndex,
    this.onReset,
    this.cameraMetadata,
  });

  @override
  State<PigInfo> createState() => _PigInfoState();
}

class _PigInfoState extends State<PigInfo> {
  final TextEditingController _distanceController = TextEditingController();
  double? _distanceMm; // stored internally in mm

  @override
  void dispose() {
    _distanceController.dispose();
    super.dispose();
  }

  /// Convert pixel length to real-world mm using camera metadata and distance.
  /// PCA measurements are diagonal (not axis-aligned), so we use the average
  /// of horizontal and vertical pixel sizes for conversion.
  double? _pixelToMm(double pixelLength) {
    final meta = widget.cameraMetadata;
    if (_distanceMm == null || meta == null) return null;

    final focalLength = meta.focalLength;
    final sensorW = meta.sensorWidth;
    final sensorH = meta.sensorHeight;
    final imgW = meta.imageWidth;
    final imgH = meta.imageHeight;

    if (focalLength == null || focalLength <= 0) return null;
    if (sensorW == null || sensorH == null || sensorW <= 0 || sensorH <= 0) return null;
    if (imgW == null || imgH == null || imgW <= 0 || imgH <= 0) return null;

    // Average pixel size since PCA axes are not aligned to image axes
    final pixelSizeW = sensorW / imgW;
    final pixelSizeH = sensorH / imgH;
    final pixelSizeMm = (pixelSizeW + pixelSizeH) / 2;

    final objectOnSensorMm = pixelLength * pixelSizeMm;
    return (objectOnSensorMm * _distanceMm!) / focalLength;
  }

  String _formatSize(double? mm) {
    if (mm == null) return '-';
    return '${(mm / 10).toStringAsFixed(1)} cm';
  }

  /// Estimate weight using regression model:
  /// Weight = -21.95431 + 0.31079(Body Length) + 0.43166(Chest Width)
  ///        + 0.47990(Abdominal Width) + 0.42656(Hip Width)
  /// All inputs in mm (converted to cm internally), output in kg.
  String _estimateWeight({
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

  @override
  Widget build(BuildContext context) {
    final hasDetections =
        widget.detectionResult != null && !widget.detectionResult!.isEmpty;

    return Padding(
      padding: ResponsiveUtils.responsivePadding(context, horizontal: 31),
      child: Container(
        width: ResponsiveUtils.width(context, 90),
        constraints: BoxConstraints(
          minHeight: ResponsiveUtils.height(context, 40),
        ),
        decoration: BoxDecoration(
          color: const Color.fromRGBO(252, 252, 252, 30),
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 13,
              spreadRadius: 6,
              offset: Offset(0, 0),
            ),
          ],
        ),
        child: Padding(
          padding:
              ResponsiveUtils.responsivePadding(context, all: 32, top: 46),
          child: hasDetections && widget.selectedPigIndex == null
              ? _buildSelectionPrompt(context)
              : hasDetections && widget.selectedPigIndex != null
                  ? _buildSelectedPigInfo(context)
                  : _buildNoDetections(context),
        ),
      ),
    );
  }

  /// Step 1: Prompt user to tap a bounding box
  Widget _buildSelectionPrompt(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'ตรวจพบหมู: ${widget.detectionResult!.count} ตัว',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 35),
            fontWeight: FontWeight.bold,
            color: Color(0xFF2671F4),
          ),
        ),
        SizedBox(height: ResponsiveUtils.height(context, 2)),
        Text(
          'แตะที่กรอบหมูเพื่อวิเคราะห์',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 28),
            color: Color(0xFF999999),
          ),
        ),
      ],
    );
  }

  /// Step 2: Show selected pig's details
  Widget _buildSelectedPigInfo(BuildContext context) {
    final det = widget.detectionResult!.detections[widget.selectedPigIndex!];
    final box = det.boundingBox;

    // Use PCA measurements from segmentation mask if available,
    // otherwise fall back to bounding box
    final PigMeasurements? pca = det.mask != null
        ? PigMeasurements.fromMask(
            det.mask!, box,
            imageWidth: widget.detectionResult!.imageWidth,
            imageHeight: widget.detectionResult!.imageHeight,
          )
        : null;

    // Pixel values: PCA-based or bounding box fallback
    final lengthPx = pca?.length ?? box.width;
    final chestPx = pca?.widthTop ?? box.height; // top width = chest
    final abdominalPx = pca?.widthMiddle ?? box.height; // middle = abdominal
    final hipPx = pca?.widthBottom ?? box.height; // bottom = hip

    // Convert to real-world mm
    final lengthMm = _pixelToMm(lengthPx);
    final chestMm = _pixelToMm(chestPx);
    final abdominalMm = _pixelToMm(abdominalPx);
    final hipMm = _pixelToMm(hipPx);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                'หมูตัวที่ ${widget.selectedPigIndex! + 1}',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 35),
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF2671F4),
                ),
              ),
            ),
            GestureDetector(
              onTap: widget.onReset,
              child: Container(
                padding: ResponsiveUtils.responsivePadding(
                  context,
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Color(0xFFEEEEEE),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'เลือกตัวอื่น',
                  style: TextStyle(
                    fontSize: ResponsiveUtils.fontSize(context, 24),
                    color: Color(0xFF5A5A5A),
                  ),
                ),
              ),
            ),
          ],
        ),
        if (AppConfig.debugMode) ...[
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          Text(
            'ความมั่นใจ: ${(det.confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: ResponsiveUtils.fontSize(context, 28),
              color: Color(0xFF5A5A5A),
            ),
          ),
        ],
        SizedBox(height: ResponsiveUtils.height(context, 2)),
        // Distance input
        _buildDistanceInput(context),
        SizedBox(height: ResponsiveUtils.height(context, 2)),
        _buildField(context, 'น้ำหนัก: ', _distanceMm != null
            ? _estimateWeight(
                bodyLengthMm: lengthMm,
                chestWidthMm: chestMm,
                abdominalWidthMm: abdominalMm,
                hipWidthMm: hipMm,
              )
            : '-'),
        if (AppConfig.debugMode) ...[
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความยาวลำตัว: ',
            _distanceMm != null
                ? _formatSize(lengthMm)
                : '${lengthPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'รอบอก (Chest): ',
            _distanceMm != null
                ? _formatSize(chestMm)
                : '${chestPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความกว้างท้อง (Abdominal): ',
            _distanceMm != null
                ? _formatSize(abdominalMm)
                : '${abdominalPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความกว้างสะโพก (Hip): ',
            _distanceMm != null
                ? _formatSize(hipMm)
                : '${hipPx.toStringAsFixed(0)} px',
          ),
        ],
      ],
    );
  }

  /// Distance input field
  Widget _buildDistanceInput(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'ระยะห่างกล้องถึงหมู (เมตร):',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 28),
            fontWeight: FontWeight.bold,
            color: Color(0xFF5A5A5A),
          ),
        ),
        SizedBox(height: ResponsiveUtils.height(context, 1)),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _distanceController,
                keyboardType: TextInputType.numberWithOptions(decimal: true),
                decoration: InputDecoration(
                  hintText: 'เช่น 0.88',
                  hintStyle: TextStyle(
                    fontSize: ResponsiveUtils.fontSize(context, 26),
                    color: Color(0xFFBBBBBB),
                  ),
                  contentPadding: EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 10,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide(color: Color(0xFFDDDDDD)),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide(color: Color(0xFF2671F4)),
                  ),
                  suffixText: 'm',
                ),
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 28),
                ),
                onSubmitted: (_) => _applyDistance(),
              ),
            ),
            SizedBox(width: 8),
            ElevatedButton(
              onPressed: _applyDistance,
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFF2671F4),
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: Text(
                'คำนวณ',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 24),
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  void _applyDistance() {
    final value = double.tryParse(_distanceController.text);
    if (value != null && value > 0) {
      setState(() {
        _distanceMm = value * 1000; // convert meters to mm
      });
      FocusScope.of(context).unfocus();
    }
  }

  /// No detections found
  Widget _buildNoDetections(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'ไม่พบหมูในรูป',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 35),
            fontWeight: FontWeight.bold,
            color: Color(0xFF999999),
          ),
        ),
        SizedBox(height: ResponsiveUtils.height(context, 2)),
        _buildField(context, 'น้ำหนัก: ', ''),
        SizedBox(height: ResponsiveUtils.height(context, 1.5)),
        _buildField(context, 'รอบอก: ', ''),
        SizedBox(height: ResponsiveUtils.height(context, 1.5)),
        _buildField(context, 'ความยาวลำตัว: ', ''),
        SizedBox(height: ResponsiveUtils.height(context, 1.5)),
        _buildField(context, 'ความกว้างลำตัว: ', ''),
      ],
    );
  }

  Widget _buildField(BuildContext context, String label, String value) {
    return Text(
      '$label$value',
      style: TextStyle(
        fontSize: ResponsiveUtils.fontSize(context, 35),
        color: Color(0xFF5A5A5A),
      ),
    );
  }
}
