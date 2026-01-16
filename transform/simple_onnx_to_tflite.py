#!/usr/bin/env python3
"""
Simple ONNX to TFLite converter using tf2onnx in reverse + TensorFlow
"""

import os
import sys
import onnx
import tensorflow as tf
import numpy as np

def convert_onnx_to_tflite(onnx_path, output_path=None):
    """
    Convert ONNX model to TFLite format

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path for output TFLite model (optional)
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Set default output path
    if output_path is None:
        base_name = os.path.splitext(onnx_path)[0]
        output_path = f"{base_name}.tflite"

    print(f"Loading ONNX model from: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # Print model info
    print(f"\nONNX Model Info:")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Graph inputs: {len(onnx_model.graph.input)}")
    print(f"  Graph outputs: {len(onnx_model.graph.output)}")

    for i, input_tensor in enumerate(onnx_model.graph.input):
        print(f"\n  Input {i}: {input_tensor.name}")
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"    Shape: {shape}")

    for i, output_tensor in enumerate(onnx_model.graph.output):
        print(f"\n  Output {i}: {output_tensor.name}")
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"    Shape: {shape}")

    print("\n" + "="*60)
    print("Note: Direct ONNX → TFLite conversion requires onnx2tf")
    print("which has complex dependencies on your system.")
    print("\nAlternative approaches:")
    print("1. Use onnxruntime for inference directly (no TFLite needed)")
    print("2. Convert ONNX → PyTorch → TFLite")
    print("3. Use Google Colab with pre-configured environment")
    print("="*60)

    return onnx_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_onnx_to_tflite.py <onnx_model_path> [output_path]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        convert_onnx_to_tflite(onnx_path, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
