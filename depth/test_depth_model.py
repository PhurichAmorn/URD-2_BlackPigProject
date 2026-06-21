"""
test_depth_model.py
-------------------
Height measurement using Depth Anything v2 Small.

Two modes:
  --metric   Uses Depth-Anything-V2-Metric-Indoor-Small-hf.
             Outputs real metres directly — no metadata needed.
             ✅ Recommended. Just run with --metric.

  (default)  Uses Depth-Anything-V2-Small-hf (relative depth).
             Outputs arbitrary units. Automatically tries to convert to
             metres using EXIF focal length + sensor size from the image.
             Falls back to "relative units" if EXIF is unavailable.

Usage:
    python test_depth_model.py --image pig.jpg --metric          # best
    python test_depth_model.py --image pig.jpg                   # tries EXIF

Interactive window: click anywhere on the depth map to print the depth at
that pixel in metres (or cm).
"""

import argparse
import time
import sys

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Path to input image")
    p.add_argument("--size", type=int, default=518,
                   help="Inference input resolution (default=518)")
    p.add_argument("--encoder", default="vits",
                   choices=["vits", "vitb", "vitl", "vitg"],
                   help="Encoder variant (vits = smallest/fastest)")
    p.add_argument("--metric", action="store_true",
                   help="Use metric-depth model (outputs real metres, no EXIF needed)")
    p.add_argument("--scene", default="indoor", choices=["indoor", "outdoor"],
                   help="Scene type for metric model (default: indoor)")
    p.add_argument("--no-show", action="store_true",
                   help="Skip interactive window, just save PNG")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EXIF — read focal length + sensor size for scale estimation
# ---------------------------------------------------------------------------
def read_exif(path: str) -> dict:
    """Return {focal_mm, sensor_w_mm, sensor_h_mm, img_w_px, img_h_px} or empty dict."""
    try:
        import exifread  # type: ignore[import-untyped]
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="EXIF FocalLength", details=False)
        result = {}

        # Focal length
        fl = tags.get("EXIF FocalLength") or tags.get("Image FocalLength")
        if fl:
            v = fl.values[0]
            result["focal_mm"] = float(v.num) / float(v.den) if hasattr(v, "num") else float(v)

        # 35mm equivalent → estimate sensor diagonal
        fl35 = tags.get("EXIF FocalLengthIn35mmFilm")
        if fl35 and "focal_mm" in result:
            eq = float(str(fl35))
            if eq > 0:
                crop = eq / result["focal_mm"]
                # 35mm full-frame diagonal ≈ 43.27 mm
                sensor_diag = 43.27 / crop
                # Assume 3:2 aspect ratio
                result["sensor_w_mm"] = sensor_diag * 3 / (3**2 + 2**2) ** 0.5
                result["sensor_h_mm"] = sensor_diag * 2 / (3**2 + 2**2) ** 0.5

        # Image dimensions
        img = Image.open(path)
        result["img_w_px"], result["img_h_px"] = img.size

        return result
    except Exception:
        return {}


def geometric_scale(depth_raw: np.ndarray, exif: dict,
                    pig_real_length_mm: float = 850.0) -> float | None:
    """
    Estimate a linear scale factor (metres per depth-unit) from EXIF camera
    parameters so that:  depth_metric_m = depth_raw * scale

    Uses: scale = (real_pig_length_mm * focal_px) / (median_pig_px * 1000)
    approximated as scale = distance_approx / median_depth_raw,
    where distance_approx comes from a rough centre-crop estimate.

    Returns None if EXIF data is insufficient.
    """
    needed = {"focal_mm", "sensor_w_mm", "img_w_px", "img_h_px"}
    if not needed.issubset(exif):
        return None

    focal_px = (exif["focal_mm"] / exif["sensor_w_mm"]) * exif["img_w_px"]
    h, w = depth_raw.shape
    centre = depth_raw[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    median_raw = float(np.median(centre))
    if median_raw == 0:
        return None

    # Treat the median relative depth as a proxy for the pig bounding-box pixel length
    # and solve for distance using the pinhole formula rearranged.
    # This is a rough heuristic — metric mode is far more accurate.
    # Assumed pixel pig length ≈ 30% of image width (typical top-down shot)
    pig_px_approx = 0.30 * w
    distance_mm = (pig_real_length_mm * focal_px) / pig_px_approx
    scale = (distance_mm / 1000.0) / median_raw
    return scale


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_ENCODER_TO_HF_NAME = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    "vitg": "Giant",
}


def load_model(encoder: str = "vits", metric: bool = False, scene: str = "indoor"):
    hf_name = _ENCODER_TO_HF_NAME.get(encoder, encoder.capitalize())
    repo = (
        f"depth-anything/Depth-Anything-V2-Metric-{scene.capitalize()}-{hf_name}-hf"
        if metric
        else f"depth-anything/Depth-Anything-V2-{hf_name}-hf"
    )
    print(f"[1/3] Loading  {repo} …")

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
        pipe = hf_pipeline(
            task="depth-estimation",
            model=repo,
            device=0 if torch.cuda.is_available() else -1,
        )
        print("      ✓ Loaded")
        return pipe, "transformers"
    except Exception as e:
        print(f"      transformers pipeline failed: {e}")

    # Manual fallback
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        from huggingface_hub import hf_hub_download

        dims = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        out_ch = [48, 96, 192, 384] if encoder == "vits" else [256, 512, 1024, 1280]
        model = DepthAnythingV2(encoder=encoder, features=dims[encoder], out_channels=out_ch)
        manual_repo = (
            f"depth-anything/Depth-Anything-V2-Metric-{scene.capitalize()}-{hf_name}"
            if metric else f"depth-anything/Depth-Anything-V2-{hf_name}"
        )
        ckpt = hf_hub_download(repo_id=manual_repo,
                               filename=f"depth_anything_v2_{encoder}.pth")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        print("      ✓ Loaded (manual)")
        return model, "manual"
    except Exception as e2:
        sys.exit(
            f"\n[ERROR] {e2}\n"
            "Run:  pip install -r requirements.txt\n"
            "Manual fallback also needs:\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
            "  pip install -e Depth-Anything-V2\n"
        )


# ---------------------------------------------------------------------------
# Inference  — returns raw depth values (metres for metric model)
# ---------------------------------------------------------------------------
def run_inference(model, mode: str, image: Image.Image, size: int):
    print("[2/3] Running inference …")
    t0 = time.perf_counter()

    if mode == "transformers":
        result = model(image)
        # "predicted_depth" is the actual tensor (metres for metric model).
        # "depth" is a PIL Image normalised 0-255 for display only — don't use it.
        raw = result["predicted_depth"]
        depth = raw.squeeze().numpy().astype(np.float32)
    else:
        depth = model.infer_image(np.array(image), size).astype(np.float32)

    elapsed = time.perf_counter() - t0
    print(f"      ✓ {elapsed * 1000:.0f} ms  shape={depth.shape}  "
          f"range=[{depth.min():.4f}, {depth.max():.4f}]")
    return depth, elapsed


# ---------------------------------------------------------------------------
# Convert to metric
# ---------------------------------------------------------------------------
def to_metric(depth_raw: np.ndarray, is_metric: bool,
              image_path: str | None) -> tuple[np.ndarray, str]:
    """
    Return (depth_m, unit_label).
    - Metric model: already in metres, pass through.
    - Relative model: try EXIF scale, else return raw with 'rel' label.
    """
    if is_metric:
        return depth_raw, "m"

    # Try EXIF-based scale
    if image_path:
        exif = read_exif(image_path)
        scale = geometric_scale(depth_raw, exif)
        if scale is not None:
            print(f"      EXIF scale factor: {scale:.4f}  "
                  f"(focal={exif.get('focal_mm', '?'):.2f} mm, "
                  f"sensor_w={exif.get('sensor_w_mm', '?'):.2f} mm)")
            return depth_raw * scale, "m (EXIF-scaled)"

    print("      ⚠  No EXIF data — depth is in relative units. Use --metric for real metres.")
    return depth_raw, "rel"


# ---------------------------------------------------------------------------
# Height stats
# ---------------------------------------------------------------------------
def print_height_stats(depth_m: np.ndarray, unit: str):
    h, w = depth_m.shape
    centre = depth_m[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

    med = float(np.median(centre))
    mn  = float(centre.min())
    mx  = float(centre.max())

    is_real = unit.startswith("m")

    def fmt(v):
        if is_real:
            return f"{v:.3f} m  ({v * 100:.1f} cm)"
        return f"{v:.4f} {unit}"

    print("\n--- Height / Distance Estimates (centre 50% crop) ---")
    print(f"  Median : {fmt(med)}  ← estimated camera height")
    print(f"  Min    : {fmt(mn)}   ← closest point (pig back)")
    print(f"  Max    : {fmt(mx)}   ← floor / far background")
    if is_real:
        print(f"\n  ➜  Camera is roughly  {med:.2f} m  ({med * 100:.0f} cm)  above the pig")
    else:
        print("\n  ➜  Add --metric for values in real metres (no EXIF/metadata needed)")


# ---------------------------------------------------------------------------
# Interactive visualisation
# ---------------------------------------------------------------------------
def visualise(image: Image.Image, depth_m: np.ndarray, elapsed: float,
              unit: str, no_show: bool):
    print("\n[3/3] Saving depth_output.png …")

    depth_norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)
    is_real = unit.startswith("m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_norm, cmap="inferno")
    axes[1].set_title(
        f"Depth map  ({elapsed * 1000:.0f} ms)  —  click to read depth",
        fontsize=9,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label(f"Depth ({unit}, normalised)")

    # Centre crop rectangle
    dh, dw = depth_m.shape
    rect = mpatches.Rectangle(
        (dw // 4, dh // 4), dw // 2, dh // 2,
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
        label="height-estimate region",
    )
    axes[1].add_patch(rect)
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.6)

    # Annotation updated on click
    annot = axes[1].annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75),
        color="white", fontsize=9,
    )
    annot.set_visible(False)

    def on_click(event):
        if event.inaxes != axes[1]:
            return
        x = int(np.clip(event.xdata + 0.5, 0, depth_m.shape[1] - 1))
        y = int(np.clip(event.ydata + 0.5, 0, depth_m.shape[0] - 1))
        val = float(depth_m[y, x])
        if is_real:
            label = f"{val:.3f} m\n({val * 100:.1f} cm)"
            print_val = f"{val:.3f} m  ({val * 100:.1f} cm)"
        else:
            label = f"{val:.4f} {unit}"
            print_val = label
        annot.xy = (x, y)
        annot.set_text(f"({x}, {y})\n{label}")
        annot.set_visible(True)
        fig.canvas.draw_idle()
        print(f"  Click ({x:4d}, {y:4d})  →  {print_val}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.suptitle(
        f"Depth Anything v2 Small  [{unit}]"
        + ("  ← real metric depth" if is_real else "  ← use --metric for real metres"),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("depth_output.png", dpi=120)
    print("      ✓ Saved depth_output.png")

    if not no_show:
        print("      Click on the depth map to measure. Close window to exit.")
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=== Depth Anything v2 — Height Measurement ===")
    if args.metric:
        print("  Mode: METRIC  (real metres, no metadata needed)\n")
    else:
        print("  Mode: relative  (will try EXIF scale; use --metric for real metres)\n")

    image = load_image(args.image)
    model, mode = load_model(args.encoder, args.metric, args.scene)
    depth_raw, elapsed = run_inference(model, mode, image, args.size)
    depth_m, unit = to_metric(depth_raw, args.metric, args.image)
    print_height_stats(depth_m, unit)
    visualise(image, depth_m, elapsed, unit, args.no_show)

    print(f"\n=== Done  ({elapsed * 1000:.0f} ms) ===")


def load_image(path: str | None) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    print("      No --image supplied — using synthetic gradient image")
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    arr[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8).reshape(-1, 1)
    arr[:, :, 2] = 128
    return Image.fromarray(arr)


if __name__ == "__main__":
    main()
