import Flutter
import UIKit
import AVFoundation

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let controller = window?.rootViewController as! FlutterViewController
    let cameraInfoChannel = FlutterMethodChannel(
      name: "camera_info",
      binaryMessenger: controller.binaryMessenger
    )
    
    cameraInfoChannel.setMethodCallHandler { (call: FlutterMethodCall, result: @escaping FlutterResult) in
      if call.method == "getCameraInfo" {
        guard let device = AVCaptureDevice.default(for: .video) else {
          result(FlutterError(
            code: "NO_CAMERA",
            message: "No camera available",
            details: nil
          ))
          return
        }
        
        // Get focal length (in mm)
        let focalLength = device.lensPosition
        
        // Get sensor dimensions from active format
        let formatDescription = device.activeFormat.formatDescription
        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
        
        // Note: iOS doesn't directly provide physical sensor size
        // We can get pixel dimensions, but physical size requires device-specific lookup
        // For now, we'll return pixel dimensions and a placeholder for physical size
        // You may need to add a device database for accurate sensor sizes
        
        let cameraInfo: [String: Any] = [
          "focalLength": focalLength,
          "sensorWidth": 0.0,  // Requires device-specific lookup
          "sensorHeight": 0.0, // Requires device-specific lookup
          "pixelWidth": Double(dimensions.width),
          "pixelHeight": Double(dimensions.height)
        ]
        
        result(cameraInfo)
      } else {
        result(FlutterMethodNotImplemented)
      }
    }
    
    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
