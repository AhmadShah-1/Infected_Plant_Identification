
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

            # Process each detected object
            for i, box in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h

                # Disregard bounding boxes larger than a fifth of the frame width
                max_box_width = frame_width // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)
                max_box_height = frame_height // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)

                if w > max_box_width or h > max_box_height:
                    continue

                mask = masks[i].astype(np.uint8) * 255  # Convert boolean mask to uint8 type
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Extract the region of interest
                roi = frame[y1:y2, x1:x2]

                # Zoom into the region of interest
                zoomed_roi = zoom_into_colored_region(roi)

                # Perform classification
                print("Start Classification")
                zoomed_pil_image = Image.fromarray(cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB))
                cls_results = cls_model(zoomed_pil_image)
                print("Completed Classification")

                # Extract the top prediction
                if cls_results:
                    names_dict = cls_results[0].names
                    probs = cls_results[0].probs.data.tolist()
                    if max(probs) >= 0.8:
                        class_name = names_dict[np.argmax(probs)]
                        confidence = max(probs)
                    else:
                        class_name = "unknown"
                        confidence = max(probs)

                    if class_name != "unknown":
                        # Draw the bounding box in green for healthy and red for infected
                        color = (0, 255, 0) if class_name == "healthy" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Save the processed image
        cv2.imwrite(output_image_path, frame)
        print(f"Processed image saved at: {output_image_path}")

print("All images processed.")
