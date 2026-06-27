import 'package:flutter/material.dart';
import 'package:DooMoo/utils/responsive.dart';
import 'package:DooMoo/models/detection_result.dart';
import 'package:DooMoo/utils/camera_metadata.dart';
import 'package:DooMoo/utils/pig_measurements.dart';
import 'package:DooMoo/utils/config.dart';
import 'package:DooMoo/utils/pig_math.dart';

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
  final TextEditingController _heightController = TextEditingController();
  double? _heightMm; // stored internally in mm
  String? _errorText;
  bool _isAutoEstimated = false;

  // Toggle for estimation reference - adjustable in debug mode
  bool _useLengthForEstimation = true;

  @override
  void dispose() {
    _heightController.dispose();
    super.dispose();
  }

  /// Convert pixel length to real-world mm using camera metadata and height (distance).
  double? _pixelToMm(double pixelLength) {
    final meta = widget.cameraMetadata;
    return PigMath.pixelToMm(
      pixelLength: pixelLength,
      distanceMm: _heightMm,
      focalLength: meta?.focalLength,
      sensorWidth: meta?.sensorWidth,
      sensorHeight: meta?.sensorHeight,
      imageWidth: meta?.imageWidth,
      imageHeight: meta?.imageHeight,
    );
  }

  String _formatSize(double? mm) {
    if (mm == null) return '-';
    return '${(mm / 10).toStringAsFixed(1)} cm';
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
          padding: ResponsiveUtils.responsivePadding(context, all: 32, top: 46),
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
            det.mask!,
            box,
            imageWidth: widget.detectionResult!.imageWidth,
            imageHeight: widget.detectionResult!.imageHeight,
            maskRect: det.maskRect,
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
        // Height input
        _buildHeightInput(context, lengthPx, chestPx),
        SizedBox(height: ResponsiveUtils.height(context, 2)),
        _buildField(
          context,
          'น้ำหนัก: ',
          _heightMm != null
              ? PigMath.estimateWeight(
                  bodyLengthMm: lengthMm,
                  chestWidthMm: chestMm,
                  abdominalWidthMm: abdominalMm,
                  hipWidthMm: hipMm,
                )
              : '-',
          fontWeight: FontWeight.bold,
        ),
        SizedBox(height: 4),
        Text(
          '* ผลการวัดเป็นเพียงค่าประมาณ น้ำหนักจริงของหมูอาจแตกต่างจากค่าที่แสดง',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 22),
            color: Color(0xFFE53935),
            fontStyle: FontStyle.italic,
          ),
        ),
        if (AppConfig.debugMode) ...[
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความยาวลำตัว: ',
            _heightMm != null
                ? _formatSize(lengthMm)
                : '${lengthPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'รอบอก (Chest): ',
            _heightMm != null
                ? _formatSize(chestMm)
                : '${chestPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความกว้างท้อง (Abdominal): ',
            _heightMm != null
                ? _formatSize(abdominalMm)
                : '${abdominalPx.toStringAsFixed(0)} px',
          ),
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          _buildField(
            context,
            'ความกว้างสะโพก (Hip): ',
            _heightMm != null
                ? _formatSize(hipMm)
                : '${hipPx.toStringAsFixed(0)} px',
          ),
        ],
      ],
    );
  }

  /// Height input field
  Widget _buildHeightInput(
      BuildContext context, double lengthPx, double chestPx) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'ความสูงจากกล้องถึงตัวหมู (เมตร):',
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 28),
            fontWeight: FontWeight.bold,
            color: Color(0xFF5A5A5A),
          ),
        ),
        SizedBox(height: ResponsiveUtils.height(context, 1)),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: TextField(
                controller: _heightController,
                keyboardType: TextInputType.numberWithOptions(decimal: true),
                decoration: InputDecoration(
                  hintText: 'เช่น 0.67',
                  hintStyle: TextStyle(
                    fontSize: ResponsiveUtils.fontSize(context, 26),
                    color: Color(0xFFBBBBBB),
                  ),
                  errorText: _errorText,
                  errorStyle: TextStyle(
                    color: Colors.red,
                    fontSize: ResponsiveUtils.fontSize(context, 22),
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
                  suffixText: 'ม.',
                ),
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 28),
                ),
                onChanged: (_) {
                  if (_errorText != null) {
                    setState(() => _errorText = null);
                  }
                },
                onSubmitted: (_) => _applyHeight(),
              ),
            ),
            SizedBox(width: 8),
            ElevatedButton(
              onPressed: _applyHeight,
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

        // Show choice only in debug mode
        if (AppConfig.debugMode) ...[
          SizedBox(height: ResponsiveUtils.height(context, 1.5)),
          Row(
            children: [
              Text(
                'ใช้ค่าอ้างอิงจาก:',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 24),
                  color: Color(0xFF5A5A5A),
                ),
              ),
              SizedBox(width: 8),
              _buildRefChip('ความยาว', true),
              SizedBox(width: 8),
              _buildRefChip('รอบอก', false),
            ],
          ),
        ],

        SizedBox(height: ResponsiveUtils.height(context, 1.5)),

        if (AppConfig.debugMode)
          // Auto-estimate button
          GestureDetector(
            onTap: () => _autoEstimateHeight(lengthPx, chestPx),
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Color(0xFFE8F0FE),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: Color(0xFF2671F4), width: 1),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.straighten, size: 16, color: Color(0xFF2671F4)),
                  SizedBox(width: 8),
                  Text(
                    'ประมาณความสูง',
                    style: TextStyle(
                      fontSize: ResponsiveUtils.fontSize(context, 24),
                      color: Color(0xFF2671F4),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),
        if (_isAutoEstimated)
          Padding(
            padding: const EdgeInsets.only(top: 6, left: 4),
            child: Text(
              _useLengthForEstimation
                  ? '* ประเมินจากความยาวเฉลี่ย (${(PigMath.averagePigLengthMm / 10).toStringAsFixed(0)} ซม.)'
                  : '* ประเมินจากความกว้างอกเฉลี่ย (${(PigMath.averagePigChestMm / 10).toStringAsFixed(1)} ซม.)',
              style: TextStyle(
                fontSize: ResponsiveUtils.fontSize(context, 22),
                color: Color(0xFFFF9800),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildRefChip(String label, bool useLength) {
    final isSelected = _useLengthForEstimation == useLength;
    return GestureDetector(
      onTap: () => setState(() => _useLengthForEstimation = useLength),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: isSelected ? Color(0xFF2671F4) : Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? Color(0xFF2671F4) : Color(0xFFDDDDDD),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: ResponsiveUtils.fontSize(context, 22),
            color: isSelected ? Colors.white : Color(0xFF5A5A5A),
          ),
        ),
      ),
    );
  }

  void _applyHeight() {
    final value = double.tryParse(_heightController.text);
    if (value != null && value > 0) {
      setState(() {
        _heightMm = value * 1000;
        _errorText = null;
        _isAutoEstimated = false;
      });
      FocusScope.of(context).unfocus();
    } else {
      setState(() {
        _errorText = 'ความสูงไม่ถูกต้อง';
        _heightMm = null;
      });
    }
  }

  void _autoEstimateHeight(double lengthPx, double chestPx) {
    final meta = widget.cameraMetadata;

    final pixelSize = _useLengthForEstimation ? lengthPx : chestPx;
    final realSizeMm = _useLengthForEstimation
        ? PigMath.averagePigLengthMm
        : PigMath.averagePigChestMm;

    final distanceMm = PigMath.estimateHeightGeometric(
      pixelSize: pixelSize,
      focalLength: meta?.focalLength,
      sensorWidth: meta?.sensorWidth,
      sensorHeight: meta?.sensorHeight,
      imageWidth: meta?.imageWidth,
      imageHeight: meta?.imageHeight,
      realSizeMm: realSizeMm,
    );

    if (distanceMm == null) {
      setState(() => _errorText = 'ไม่มีข้อมูลกล้อง กรุณาใส่ระยะด้วยตนเอง');
      return;
    }

    final distanceM = distanceMm / 1000.0;

    if (AppConfig.debugMode) {
      print('--- Height Estimation Debug ---');
      print('Reference: ${_useLengthForEstimation ? "Length" : "Chest"}');
      print('Pixel Size: $pixelSize px');
      print('Real Size: $realSizeMm mm');
      print('Estimated Height: ${distanceM.toStringAsFixed(6)} m');
    }

    setState(() {
      _heightMm = distanceMm;
      _heightController.text = distanceM.toStringAsFixed(3);
      _errorText = null;
      _isAutoEstimated = true;
    });
    FocusScope.of(context).unfocus();
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

  Widget _buildField(BuildContext context, String label, String value,
      {FontWeight fontWeight = FontWeight.normal}) {
    return Text(
      '$label$value',
      style: TextStyle(
        fontSize: ResponsiveUtils.fontSize(context, 35),
        color: Color(0xFF5A5A5A),
        fontWeight: fontWeight,
      ),
    );
  }
}
