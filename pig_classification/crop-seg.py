import cv2
from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from rfdetr import RFDETRSegPreview

yolo_model = YOLO('best.pt')
image_path = '../../images/omkoi8.jpg'
img = cv2.imread(image_path)
results = yolo_model(image_path)[0]

boxes = results.boxes.xyxy.cpu().numpy()
selected_crop = None
selected_box_idx = None

device = torch.device('cpu')
print("Using device:", device)

print("Loading RF-DETR segmentation model...")
rfdetr_model = RFDETRSegPreview(
    pretrain_weights='../pig_segmentation/best_pig_segmentation_model.pth',
    model_size='n'
)
print("RF-DETR model loaded successfully!")

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225])
])

def run_rfdetr_segmentation(crop_img, box):
    """Run RF-DETR segmentation on the cropped pig image."""
    print("Running RF-DETR segmentation...")
    
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    try:
        detections = rfdetr_model.predict(crop_rgb, threshold=0.25)
        print(f"Detected {len(detections)} objects")
        
        if detections.mask is not None and len(detections.mask) > 0:
            combined_mask = np.zeros((crop_img.shape[0], crop_img.shape[1]), dtype=np.uint8)
            for mask in detections.mask:
                combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8))
            
            mask_resized = combined_mask.astype(np.float32)
            
            masked_pig = crop_img.copy()
            masked_pig[mask_resized < 0.5] = [255, 255, 255]
            
            cv2.imshow("RF-DETR Segmentation Mask", (mask_resized * 255).astype(np.uint8))
            cv2.imshow("Masked Pig", masked_pig)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Cropped Pig (YOLO)")
            axes[0].axis("off")
            
            axes[1].imshow(mask_resized, cmap='gray')
            axes[1].set_title("RF-DETR Segmentation Mask")
            axes[1].axis("off")
            
            axes[2].imshow(cv2.cvtColor(masked_pig, cv2.COLOR_BGR2RGB))
            axes[2].set_title("Masked Pig")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.show()
            
            print("RF-DETR segmentation complete!")
        else:
            print("Warning: No mask detected from RF-DETR")
            cv2.imshow("Selected Crop (YOLO)", crop_img)
    except Exception as e:
        print(f"Error during segmentation: {e}")
        cv2.imshow("Selected Crop (YOLO)", crop_img)

def select_box(event, x, y, flags, param):
    global selected_crop, selected_box_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Selected Pig #{i+1}")
                selected_crop = img[y1:y2, x1:x2]
                selected_box_idx = i
                run_rfdetr_segmentation(selected_crop, box)
                break

cv2.namedWindow("Click a box to crop")
cv2.setMouseCallback("Click a box to crop", select_box)

display_img = results.plot()

print("Instructions: Left-click inside any green box to select that pig. Press 'q' to exit.")

while True:
    cv2.imshow("Click a box to crop", display_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()