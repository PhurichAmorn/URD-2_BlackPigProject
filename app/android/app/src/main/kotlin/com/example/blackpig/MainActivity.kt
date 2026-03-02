package com.example.blackpig

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.SizeF
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "camera_info"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "getCameraInfo") {
                try {
                    val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
                    val cameraIds = cameraManager.cameraIdList
                    
                    if (cameraIds.isEmpty()) {
                        result.error("NO_CAMERA", "No camera available", null)
                        return@setMethodCallHandler
                    }
                    
                    // Use the first available camera (usually back camera)
                    val cameraId = cameraIds[0]
                    val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                    
                    // Get focal length
                    val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    val focalLength = if (focalLengths != null && focalLengths.isNotEmpty()) {
                        focalLengths[0].toDouble()
                    } else {
                        0.0
                    }
                    
                    // Get sensor physical size (in mm)
                    val sensorSize: SizeF? = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    val sensorWidth = sensorSize?.width?.toDouble() ?: 0.0
                    val sensorHeight = sensorSize?.height?.toDouble() ?: 0.0
                    
                    val cameraInfo = mapOf(
                        "focalLength" to focalLength,
                        "sensorWidth" to sensorWidth,
                        "sensorHeight" to sensorHeight
                    )
                    
                    result.success(cameraInfo)
                } catch (e: Exception) {
                    result.error("ERROR", "Failed to get camera info: ${e.message}", null)
                }
            } else {
                result.notImplemented()
            }
        }
    }
}
