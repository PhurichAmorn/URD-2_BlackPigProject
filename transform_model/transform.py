"""
transform.py
Convert PyTorch segment_model.pth to TensorFlow Lite format

This script converts a DeepLabV3-ResNet101 segmentation model from PyTorch to TensorFlow Lite
using the onnx2tf library for better compatibility.

Requirements:
    uv pip install torch torchvision onnx onnx2tf tensorflow onnxruntime

Author: Claude Code
Date: 13 November 2025
"""

import torch
import onnx
import tensorflow as tf
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import os
import subprocess
import sys


def convert_pytorch_to_tflite(
    pytorch_model_path='segment_model.pth',
    output_dir='.',
    input_shape=(1, 3, 256, 256),
    num_classes=1,
    quantize=False
):
    """
    Convert PyTorch segmentation model to TensorFlow Lite using onnx2tf.

    Args:
        pytorch_model_path (str): Path to PyTorch .pth model file
        output_dir (str): Directory to save converted models
        input_shape (tuple): Input shape (batch, channels, height, width)
        num_classes (int): Number of output classes
        quantize (bool): Whether to apply quantization for smaller model size

    Returns:
        str: Path to the generated TFLite model
    """

    print("=" * 60)
    print("PyTorch to TensorFlow Lite Conversion")
    print("=" * 60)

    # Step 1: Load PyTorch model
    print("\n[1/4] Loading PyTorch model...")
    device = torch.device('cpu')  # Use CPU for conversion

    # Create model with no pretrained weights (weights=None prevents downloading backbone)
    model = deeplabv3_resnet101(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        aux_loss=True
    )

    # Load the custom trained weights
    model.load_state_dict(torch.load(
        pytorch_model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"✓ Loaded model from {pytorch_model_path}")

    # Step 2: Export to ONNX
    print("\n[2/4] Exporting to ONNX format...")
    onnx_path = os.path.join(output_dir, 'segment_model.onnx')

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Use opset 18 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ Exported to ONNX: {onnx_path}")

    # Verify ONNX model
    print("\n[3/4] Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Step 3: Convert ONNX to TensorFlow using onnx2tf Python API
    print("\n[4/4] Converting ONNX to TensorFlow and TFLite...")
    tf_model_dir = os.path.join(output_dir, 'segment_model_tf')

    try:
        import onnx2tf

        # Use Python API instead of CLI to avoid library loading issues
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_model_dir,
            copy_onnx_input_output_names_to_tflite=True,
            output_signaturedefs=True,
            output_integer_quantized_tflite=quantize,
            quant_type='per-tensor' if quantize else None,
        )

        print("✓ Conversion completed successfully")

        # Find the generated TFLite file
        tflite_files = [f for f in os.listdir(
            tf_model_dir) if f.endswith('.tflite')]

        if not tflite_files:
            raise FileNotFoundError("No .tflite file was generated")

        # Use the float32 model (non-quantized) or int8 (quantized)
        if quantize:
            tflite_file = next(
                (f for f in tflite_files if 'int8' in f.lower() or 'quantized' in f.lower()),
                tflite_files[0])
        else:
            tflite_file = next(
                (f for f in tflite_files if 'float32' in f.lower() or 'float' in f.lower()),
                tflite_files[0])

        # Copy to desired location
        src_tflite = os.path.join(tf_model_dir, tflite_file)
        tflite_filename = 'segment_model_quantized.tflite' if quantize else 'segment_model.tflite'
        dest_tflite = os.path.join(output_dir, tflite_filename)

        import shutil
        shutil.copy2(src_tflite, dest_tflite)
        tflite_path = dest_tflite

    except Exception as e:
        print(f"\n✗ Error during onnx2tf conversion:")
        print(f"Error: {e}")
        print("\nNote: ONNX model was successfully created at:", onnx_path)
        print("You can manually convert it using online tools or other converters.")
        raise

    print(f"✓ TensorFlow Lite model saved: {tflite_path}")

    # Print model info
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Input shape: {input_shape} (NCHW format - PyTorch)")
    print(f"Output classes: {num_classes}")
    print(f"Quantized: {quantize}")

    # Get file sizes
    pth_size = os.path.getsize(pytorch_model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

    print(f"\nFile sizes:")
    print(f"  PyTorch model:     {pth_size:.2f} MB")
    print(f"  TFLite model:      {tflite_size:.2f} MB")
    print(f"  Compression ratio: {pth_size/tflite_size:.2f}x")

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)

    return tflite_path


def test_tflite_model(tflite_path, test_image_shape=(256, 256, 3)):
    """
    Test the TFLite model with a random input.

    Args:
        tflite_path (str): Path to TFLite model
        test_image_shape (tuple): Test image shape (height, width, channels)
    """
    print("\n" + "=" * 60)
    print("Testing TensorFlow Lite Model")
    print("=" * 60)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel Details:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    # Create test input based on the model's expected input shape
    expected_shape = input_details[0]['shape']
    test_input = np.random.randn(*expected_shape).astype(np.float32)

    # Normalize like in the original code (if needed)
    if len(expected_shape) == 4 and expected_shape[-1] == 3:
        # NHWC format
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    else:
        # Just use random normalized data
        mean = 0
        std = 1

    test_input = (test_input - mean) / std

    print("\nRunning inference...")

    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        print(f"✓ Inference successful!")
        print(f"  Output shape: {output_data.shape}")
        print(
            f"  Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise

    print("\n" + "=" * 60)


def main():
    """Main entry point for the conversion script."""
    # Configuration
    PYTORCH_MODEL_PATH = 'segment_model.pth'
    OUTPUT_DIR = '.'
    INPUT_SHAPE = (1, 3, 256, 256)  # (batch, channels, height, width)
    NUM_CLASSES = 1
    # Set to True for smaller model size (with slight accuracy loss)
    QUANTIZE = False

    # Check if PyTorch model exists
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print(f"Error: PyTorch model not found at {PYTORCH_MODEL_PATH}")
        print("Please ensure the model file exists in the current directory.")
        sys.exit(1)

    # Convert model
    try:
        tflite_path = convert_pytorch_to_tflite(
            pytorch_model_path=PYTORCH_MODEL_PATH,
            output_dir=OUTPUT_DIR,
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            quantize=QUANTIZE
        )

        # Test the converted model
        test_tflite_model(tflite_path)

    except Exception as e:
        print(f"\n✗ Error during conversion: {str(e)}")
        print("\nTroubleshooting tips:")
        print(
            "1. Install required packages: uv pip install torch torchvision onnx onnx2tf tensorflow onnxruntime")
        print("2. Ensure you have enough disk space")
        print("3. Try setting QUANTIZE=False if quantization fails")
        print("4. Check that onnx2tf is installed: onnx2tf --help")
        raise


if __name__ == "__main__":
    main()
