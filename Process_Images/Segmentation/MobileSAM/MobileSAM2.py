# This version applies Non Maximum Suppression to remvoe duplicate bounding boxes

import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry as mobile_sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Segmentation Model
model_type = "vit_t"
sam_checkpoint = "C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/weights/segmentation/MobileSAM1/weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Print a statement if using CPU
if device == "cpu":
    print("Warning: The model is using the CPU. This may slow down processing time.")

mobile_sam = mobile_sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)

# Load the classification model
cls_model = YOLO('C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/weights/classification/classification1/last.pt')

# Define the input, mask, and output directories
input_dir = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Far_Images/images'  # Replace with your input directory path
mask_dir = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Random/MobileSAM/mask'
output_dir = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Random/MobileSAM/output'
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Function to zoom into the region of interest based on colored pixels
def zoom_into_colored_region(image, zoom_factor=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    roi = image[y:y + h, x:x + w]

    new_width = int(w * zoom_factor)
    new_height = int(h * zoom_factor)
    zoomed_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return zoomed_roi

# Non-Maximum Suppression (NMS) function
def non_max_suppression(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    # Convert to numpy arrays for easier indexing
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort by scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# Process each image in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if needed
        image_path = os.path.join(input_dir, filename)
        mask_output_path = os.path.join(mask_dir, f"mask_{filename}")
        output_image_path = os.path.join(output_dir, f"processed_{filename}")

        # Read the image
        frame = cv2.imread(image_path)
        frame_height, frame_width, _ = frame.shape

        # Resize the frame to make it smaller
        frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

        # Perform SAM segmentation
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Starting Segmentation Analysis for {filename}")
        with torch.no_grad():  # Disable gradient calculation for inference
            sam_result = mask_generator.generate(image_rgb)
        print("Completed Segmentation Analysis")

        # Check if masks were generated
        if sam_result:
            masks = [mask['segmentation'] for mask in sam_result]
            boxes = [mask['bbox'] for mask in sam_result]

            # Combine all masks into one image
            combined_mask = np.zeros_like(frame[:, :, 0])
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask.astype(np.uint8) * 255)

            # Save the combined mask
            cv2.imwrite(mask_output_path, combined_mask)
            print(f"Mask saved at: {mask_output_path}")

            # Collect boxes and scores for NMS
            boxes_for_nms = []
            scores_for_nms = []
            class_names = []

            for i, box in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h

                # Disregard bounding boxes larger than a fifth of the frame width
                max_box_width = frame_width // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)
                max_box_height = frame_height // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)

                if w > max_box_width or h > max_box_height:
                    continue

                roi = frame[y1:y2, x1:x2]
                zoomed_roi = zoom_into_colored_region(roi)
                zoomed_pil_image = Image.fromarray(cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB))

                cls_results = cls_model(zoomed_pil_image)
                if cls_results:
                    names_dict = cls_results[0].names
                    probs = cls_results[0].probs.data.tolist()
                    max_prob = max(probs)
                    if max_prob >= 0.8:
                        class_name = names_dict[np.argmax(probs)]
                        boxes_for_nms.append([x1, y1, x2, y2])
                        scores_for_nms.append(max_prob)
                        class_names.append(class_name)

            # Apply NMS
            keep_indices = non_max_suppression(boxes_for_nms, scores_for_nms, iou_threshold=0.5)

            # Draw the bounding boxes
            for idx in keep_indices:
                x1, y1, x2, y2 = boxes_for_nms[idx]
                class_name = class_names[idx]
                color = (0, 255, 0) if class_name == "healthy" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Save the processed image
        cv2.imwrite(output_image_path, frame)
        print(f"Processed image saved at: {output_image_path}")

print("All images processed.")
