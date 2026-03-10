import 'package:flutter/material.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/utils/camera_metadata.dart';

class CameraMetadataInfo extends StatelessWidget {
  final CameraMetadata? cameraMetadata;

  const CameraMetadataInfo({super.key, this.cameraMetadata});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: ResponsiveUtils.responsivePadding(context, horizontal: 31),
      child: Container(
        width: ResponsiveUtils.width(context, 90),
        constraints: BoxConstraints(
          minHeight: ResponsiveUtils.height(context, 30),
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
          padding: ResponsiveUtils.responsivePadding(context, all: 20, top: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Camera Metadata',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 32),
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF3A3A3A),
                ),
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              _buildMetadataRow(
                context,
                'Focal Length',
                cameraMetadata?.focalLength != null
                    ? '${cameraMetadata!.focalLength!.toStringAsFixed(2)} mm'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'Sensor Width',
                cameraMetadata?.sensorWidth != null
                    ? '${cameraMetadata!.sensorWidth!.toStringAsFixed(2)} mm'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'Sensor Height',
                cameraMetadata?.sensorHeight != null
                    ? '${cameraMetadata!.sensorHeight!.toStringAsFixed(2)} mm'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'Image Width',
                cameraMetadata?.imageWidth != null
                    ? '${cameraMetadata!.imageWidth} px'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'Image Height',
                cameraMetadata?.imageHeight != null
                    ? '${cameraMetadata!.imageHeight} px'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'F-Number',
                cameraMetadata?.fNumber != null
                    ? 'f/${cameraMetadata!.fNumber!.toStringAsFixed(1)}'
                    : 'N/A',
              ),
              _buildMetadataRow(
                context,
                'ISO',
                cameraMetadata?.iso != null ? '${cameraMetadata!.iso}' : 'N/A',
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMetadataRow(BuildContext context, String label, String value) {
    // Use smaller font size to ensure text fits in the box
    final fontSize = ResponsiveUtils.fontSize(context, 26);

    return Padding(
      padding:
          EdgeInsets.symmetric(vertical: ResponsiveUtils.height(context, 0.5)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Flexible(
            flex: 1,
            child: Text(
              label,
              style: TextStyle(
                fontSize: fontSize,
                color: Color(0xFF5A5A5A),
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          SizedBox(width: ResponsiveUtils.width(context, 2)),
          Flexible(
            flex: 1,
            child: Text(
              value,
              style: TextStyle(
                fontSize: fontSize,
                fontWeight: FontWeight.w500,
                color: Color(0xFF3A3A3A),
              ),
              textAlign: TextAlign.end,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
