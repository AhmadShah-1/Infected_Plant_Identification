# Takes approximately 3-4 minutes to process a frame

import os
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from ultralytics import YOLO

# Define paths
HOME = "C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/weights/segmentation/SAM1"
CHECKPOINT_PATH = os.path.join(HOME, "../weights", "sam_vit_h_4b8939.pth")

# Load the SAM model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Load the classification model
cls_model = YOLO('C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/weights/classification/classification1/last.pt')  # Classification model path

# Open the video stream
video_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Video/vid1.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(fps)

# Define the codec and create VideoWriter object
output_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Output_Video/Process_SAM1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width // 2, frame_height // 2))

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


frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to make it smaller
    frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    frame_count += 1

    # Process every 29th frame
    if frame_count > 200 and frame_count % 100 == 0:
        # Perform SAM segmentation
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)

        # Check if masks were generated
        if sam_result:
            masks = [mask['segmentation'] for mask in sam_result]
            boxes = [mask['bbox'] for mask in sam_result]

            # Process each detected object
            for i, box in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h
                mask = masks[i].astype(np.uint8) * 255  # Convert boolean mask to uint8 type
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Extract the region of interest
                roi = frame[y1:y2, x1:x2]

                # Zoom into the region of interest
                zoomed_roi = zoom_into_colored_region(roi)

                # Perform classification
                zoomed_pil_image = Image.fromarray(cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB))
                cls_results = cls_model(zoomed_pil_image)

                # Extract the top prediction
                names_dict = cls_results[0].names
                probs = cls_results[0].probs.data.tolist()
                class_name = names_dict[np.argmax(probs)] if max(probs) >= 0.8 else "unknown"
                confidence = max(probs)

                # Draw the bounding box and class label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

    # Save the processed frame
    out.write(frame)

    # Display the frame with predictions using OpenCV
    cv2.imshow('Processed Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Processed video saved at:", output_path)
