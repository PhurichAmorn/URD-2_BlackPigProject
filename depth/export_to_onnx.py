"""
export_to_onnx.py
-----------------
Export Depth Anything v2 Small to ONNX with optional INT8 quantization.

Steps performed:
  1. Load DA-v2-Small weights from HuggingFace Hub.
  2. Trace/export to ONNX (opset 17, dynamic batch).
  3. (Optional) Apply ONNX INT8 static quantization via onnxruntime.
  4. Run a quick sanity-check inference on both models.
  5. Print model sizes.

Output files (written to ./models/):
  depth_anything_v2_small.onnx           ~100 MB FP32
  depth_anything_v2_small_int8.onnx      ~25  MB INT8  (if --quantize)

Usage:
    python export_to_onnx.py
    python export_to_onnx.py --quantize
    python export_to_onnx.py --input-size 256 --quantize
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square input resolution for the exported model (default=518; use 256 for lightest mobile variant)",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 static quantization after ONNX export",
    )
    p.add_argument(
        "--calibration-images",
        nargs="*",
        default=None,
        metavar="IMAGE",
        help="Images used for INT8 calibration (optional; random tensors used if omitted)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_ENCODER_TO_HF_NAME = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    "vitg": "Giant",
}


def load_pytorch_model(encoder: str = "vits"):
    """Return (model, device) for the requested encoder variant."""
    import torch

    hf_name = _ENCODER_TO_HF_NAME.get(encoder, encoder.capitalize())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Try transformers-style load first (no extra repo needed)
    try:
        from transformers import AutoModelForDepthEstimation

        model = AutoModelForDepthEstimation.from_pretrained(
            f"depth-anything/Depth-Anything-V2-{hf_name}-hf"
        ).to(device).eval()
        print("  Loaded via transformers AutoModel")
        return model, device, "transformers"
    except Exception as e:
        print(f"  transformers load failed ({e}); trying manual load …")

    # Manual: requires cloning Depth-Anything-V2 and `pip install -e .`
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download
        import torch

        model = DepthAnythingV2(encoder=encoder, features=384, out_channels=[48, 96, 192, 384])
        ckpt = hf_hub_download(
            repo_id=f"depth-anything/Depth-Anything-V2-{hf_name}",
            filename=f"depth_anything_v2_{encoder}.pth",
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        print("  Loaded via depth_anything_v2 package")
        return model, device, "manual"
    except Exception as e2:
        sys.exit(
            f"\n[ERROR] Could not load model: {e2}\n"
            "Install:  pip install -r requirements.txt\n"
            "Manual fallback also needs:\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
            "  pip install -e Depth-Anything-V2\n"
        )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(model, mode: str, device: str, input_size: int) -> Path:
    import torch

    onnx_path = OUT_DIR / "depth_anything_v2_small.onnx"
    print(f"\n[2/4] Exporting to ONNX → {onnx_path}")

    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    if mode == "transformers":
        # Wrap forward so ONNX sees a simple tensor→tensor graph
        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                out = self.m(pixel_values=pixel_values)
                return out.predicted_depth  # (1, H, W)

        wrapped = _Wrapper(model).eval()
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(onnx_path),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["depth"],
            dynamic_axes={"pixel_values": {0: "batch"}, "depth": {0: "batch"}},
            do_constant_folding=True,
        )
    else:
        # Manual model — infer_image not trace-friendly; export forward directly
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            opset_version=17,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={"image": {0: "batch"}, "depth": {0: "batch"}},
            do_constant_folding=True,
        )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  ✓ Saved  {onnx_path.name}  ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------
def validate_onnx(onnx_path: Path, input_size: int):
    import onnxruntime as ort

    print(f"\n[3/4] Validating {onnx_path.name} …")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    t0 = time.perf_counter()
    out = sess.run(None, {inp_name: dummy})
    elapsed = (time.perf_counter() - t0) * 1000

    depth = out[0]
    print(f"  Output shape : {depth.shape}")
    print(f"  Depth range  : [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Latency (CPU): {elapsed:.0f} ms")
    return elapsed


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------
def quantize_int8(onnx_path: Path, calibration_images, input_size: int) -> Path:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

    int8_path = OUT_DIR / "depth_anything_v2_small_int8.onnx"
    print(f"\n[4/4] Applying INT8 quantization → {int8_path}")

    inp_name = "pixel_values"  # adjust if export used "image"

    class _Reader(CalibrationDataReader):
        def __init__(self):
            if calibration_images:
                from PIL import Image

                self._data = []
                for p in calibration_images:
                    img = Image.open(p).convert("RGB").resize((input_size, input_size))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    # ImageNet normalisation
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    arr = (arr - mean) / std
                    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)
                    self._data.append({inp_name: arr})
            else:
                # Random calibration — less accurate but works for smoke-test
                print("  No calibration images supplied — using random tensors")
                self._data = [
                    {inp_name: np.random.rand(1, 3, input_size, input_size).astype(np.float32)}
                    for _ in range(20)
                ]
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    quantize_static(
        str(onnx_path),
        str(int8_path),
        calibration_data_reader=_Reader(),
        quant_type=QuantType.QInt8,
    )
    size_mb = int8_path.stat().st_size / 1e6
    print(f"  ✓ Saved  {int8_path.name}  ({size_mb:.1f} MB)")
    return int8_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=== Depth Anything v2 Small — ONNX Export ===")
    print(f"  Input size : {args.input_size}×{args.input_size}")
    print(f"  Quantize   : {args.quantize}\n")

    print("[1/4] Loading PyTorch model …")
    model, device, mode = load_pytorch_model("vits")

    onnx_path = export_onnx(model, mode, device, args.input_size)
    validate_onnx(onnx_path, args.input_size)

    if args.quantize:
        int8_path = quantize_int8(onnx_path, args.calibration_images, args.input_size)
        print("\n  Validating INT8 model …")
        validate_onnx(int8_path, args.input_size)

    print("\n=== Done ===")
    print(f"  FP32 model : {onnx_path}")
    if args.quantize:
        print(f"  INT8 model : {int8_path}")
    print(
        "\nNext step: copy the INT8 model into the Flutter app:\n"
        "  DooMoo/assets/models/depth_anything_v2_small_int8.onnx\n"
        "and update DooMoo/pubspec.yaml assets list."
    )


if __name__ == "__main__":
    main()
