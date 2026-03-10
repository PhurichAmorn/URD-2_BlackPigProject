"""
Export RF-DETR segmentation model to ONNX format for on-device inference.

Usage:
    python scripts/export_onnx.py

Prerequisites:
    pip install rfdetr onnx onnxconverter-common

After export, open the .onnx file in Netron (https://netron.app/) to confirm:
  - Input tensor name + shape (e.g., images: float32[1, 3, H, W])
  - Output tensor names + shapes (e.g., dets, labels, masks)
Then update the tensor names in lib/services/pig_detector.dart accordingly.
"""

import os
from rfdetr import RFDETRSegPreview

CHECKPOINT_PATH = "/Users/nine/Desktop/CMKL/2.2/urd/checkpoint_best_total.pth"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rf_detr_pig.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = RFDETRSegPreview(pretrain_weights=CHECKPOINT_PATH)

# Print the resolution the model expects
resolution = getattr(model, "resolution", None)
print(f"Model resolution: {resolution}")

model.export(
    output_path=OUTPUT_PATH,
    segmentation_head=True,
    simplify=False,
)

print(f"Exported ONNX model to: {OUTPUT_PATH}")

# Optional: FP16 quantization (~65 MB instead of ~130 MB)
# Uncomment the lines below to apply FP16 quantization:
#
# from onnxconverter_common import float16
# import onnx
# onnx_model = onnx.load(OUTPUT_PATH)
# model_fp16 = float16.convert_float_to_float16(onnx_model)
# onnx.save(model_fp16, OUTPUT_PATH)
# print(f"Applied FP16 quantization to: {OUTPUT_PATH}")
