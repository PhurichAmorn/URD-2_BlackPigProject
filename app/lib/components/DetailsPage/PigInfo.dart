import 'package:flutter/material.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/models/detection_result.dart';

class PigInfo extends StatelessWidget {
  final DetectionResult? detectionResult;

  const PigInfo({super.key, this.detectionResult});

  @override
  Widget build(BuildContext context) {
    final hasDetections =
        detectionResult != null && !detectionResult!.isEmpty;

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
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Detection summary
              if (hasDetections) ...[
                Text(
                  'ตรวจพบหมู: ${detectionResult!.count} ตัว',
                  style: TextStyle(
                    fontSize: ResponsiveUtils.fontSize(context, 35),
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2671F4),
                  ),
                ),
                SizedBox(height: ResponsiveUtils.height(context, 1.5)),
                // Per-detection confidence
                ...detectionResult!.detections.asMap().entries.map((entry) {
                  final i = entry.key;
                  final det = entry.value;
                  return Padding(
                    padding: EdgeInsets.only(
                        bottom: ResponsiveUtils.height(context, 0.5)),
                    child: Text(
                      'หมูตัวที่ ${i + 1}: ความมั่นใจ ${(det.confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: ResponsiveUtils.fontSize(context, 28),
                        color: Color(0xFF5A5A5A),
                      ),
                    ),
                  );
                }),
                SizedBox(height: ResponsiveUtils.height(context, 2)),
              ] else ...[
                Text(
                  'ไม่พบหมูในรูป',
                  style: TextStyle(
                    fontSize: ResponsiveUtils.fontSize(context, 35),
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF999999),
                  ),
                ),
                SizedBox(height: ResponsiveUtils.height(context, 2)),
              ],

              // Measurement fields (proxy data from bounding box)
              _buildField(
                context,
                'น้ำหนัก: ',
                hasDetections ? '-' : '',
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              _buildField(
                context,
                'รอบอก: ',
                hasDetections
                    ? '${_primaryBox?.height.toStringAsFixed(0)} px'
                    : '',
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              _buildField(
                context,
                'ความยาวลำตัว: ',
                hasDetections
                    ? '${_primaryBox?.width.toStringAsFixed(0)} px'
                    : '',
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              _buildField(
                context,
                'ความกว้างลำตัว: ',
                hasDetections
                    ? '${_primaryBox?.height.toStringAsFixed(0)} px'
                    : '',
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Primary (highest confidence) detection bounding box, if any.
  Rect? get _primaryBox {
    if (detectionResult == null || detectionResult!.isEmpty) return null;
    return detectionResult!.detections.first.boundingBox;
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
