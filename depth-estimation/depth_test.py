from PIL import Image
import cv2
import numpy as np
# from ultralytics import YOLO
from pathlib import Path
import depth_pro
import matplotlib.pyplot as plt
import torch
import os

# yolo_model = YOLO("yolo11s.pt")

# --- Load and resize image ---
image_path = "../../../images/height0.838m.jpg" # 1.43 0.75
image_pil = Image.open(image_path).convert("RGB")
image_pil.thumbnail((640, 640))
image = np.array(image_pil)

# # run yolo
# results = yolo_model(image)

# prediction_boxes = []
# names = yolo_model.names

# for result in results:
#     annotated = result.plot()  # draw boxes and labels on image
#     for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
#         if int(cls) == 19:  # 0 = human, 16 = dog, 19 = cow
#             prediction_boxes.append(box.cpu().numpy())

# yolo_output_path = "output/yolo_cow_predictions.jpg"
# cv2.imwrite(yolo_output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

# Run depth pro
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

if torch.cuda.is_available():
    depth_model = depth_model.cuda()

image_rgb, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image_rgb)

if torch.cuda.is_available():
    depth_input = {k: v.cuda() for k, v in depth_input.items()}

# Actual focal length in pixel from metadata
# Formula focal length (px) = focal_length (mm) * image width or height / sensor width or height
actual_fl_px =  2729# iphone 15 pm 5293.380822/3970 xiaomi = 2729.166667

# convert to tensor (depth_pro.infer expects tensor/array with .squeeze())
f_px_t = torch.tensor(actual_fl_px, dtype=torch.float32)
if torch.cuda.is_available():
    f_px_t = f_px_t.cuda()

prediction = depth_model.infer(depth_input, f_px=f_px_t)
depth = prediction["depth"]
depth_np = depth.squeeze().cpu().numpy()

depth_output_path = "output/depth_map.png"
os.makedirs(os.path.dirname(depth_output_path), exist_ok=True)

# Interactive plot: click to read depth at a pixel
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(depth_np, cmap="plasma")
ax.set_title("Depth Map (meters) — click a point to read distance")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")

# annotation and marker that will update on clicks
annot = ax.text(0.02, 0.96, "", color="white", transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor="black", alpha=0.6))
marker = ax.scatter([], [], s=70, c="cyan", edgecolors="black", linewidths=0.8)

def onclick(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    h, w = depth_np.shape
    if not (0 <= row < h and 0 <= col < w):
        print("Click outside image bounds")
        return
    depth_val = depth_np[row, col]
    if np.isnan(depth_val):
        msg = f"Depth at ({col},{row}): NaN"
    else:
        msg = f"Depth at ({col},{row}): {depth_val:.3f} m"
    print(msg)
    # update visual feedback
    annot.set_text(msg)
    marker.set_offsets(np.array([[col, row]]))
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

# Save a static copy and show interactive window
plt.savefig(depth_output_path, bbox_inches="tight")
print(f"Depth map saved as: {depth_output_path}")
plt.show()

# disconnect handler after window closes
fig.canvas.mpl_disconnect(cid)

plt.figure(figsize=(10, 8))
plt.imshow(depth_np, cmap="plasma")
plt.title("Depth Map (meters)")
plt.colorbar(label="Depth (m)")
plt.axis("off")
plt.savefig(depth_output_path, bbox_inches="tight")


# if not prediction_boxes:
#     print("No object detected!")
# else:
#     for i, box in enumerate(prediction_boxes):
#         x1, y1, x2, y2 = box.astype(int)
#         region = depth_np[y1:y2, x1:x2]
#         if region.size == 0:
#             continue
#         mean_depth = np.mean(region)
#         min_depth = np.min(region)
#         max_depth = np.max(region)
#         print(f"Dog {i}: mean = {mean_depth:.2f} m, min = {min_depth:.2f} m, max = {max_depth:.2f} m")

# print(f"YOLO prediction saved as: {yolo_output_path}")
print(f"Depth map saved as: {depth_output_path}")
