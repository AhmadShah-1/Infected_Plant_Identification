import os
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry as original_sam_model_registry, \
    SamAutomaticMaskGenerator as OriginalSamAutomaticMaskGenerator
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

# Open the video stream
video_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Video/vid1.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"FPS: {fps}")

# Define the codec and create VideoWriter object
output_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Output_Video/Process_SAM1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width // 2, frame_height // 2))

# Define the directory to save processed frames
processed_frames_dir = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Output_Images/'
os.makedirs(processed_frames_dir, exist_ok=True)


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
max_box_width = frame_width // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)
max_box_height = frame_height // 10  # A fifth of the frame width divided by 2 (because frame is resized by half)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to make it smaller
    frame = cv2.resize(frame, (frame_width // 4, frame_height // 4))

    frame_count += 1

    # Process every 200th frame after the 900th frame
    if frame_count % 30 == 0:

        '''
        # Ask the user if they want to run inference on this frame
        user_input = input(f"Process frame {frame_count}? (y/n): ").strip().lower()
        if user_input != 'y':
            continue
            
        '''

        # Perform SAM segmentation
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Starting Segmentation Analysis")
        with torch.no_grad():  # Disable gradient calculation for inference
            sam_result = mask_generator.generate(image_rgb)
        print("Completed Segmentation Analysis")

        # Check if masks were generated
        if sam_result:
            masks = [mask['segmentation'] for mask in sam_result]
            boxes = [mask['bbox'] for mask in sam_result]

            # Process each detected object
            for i, box in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h

                # Disregard bounding boxes larger than a fifth of the frame width
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
                        # Draw the bounding box and class label on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 255, 0), 2)

            # Save the processed frame
            frame_filename = os.path.join(processed_frames_dir, f"frame_bboxLimit_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)

    # Save the processed frame to video
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
print(f"Processed frames saved at: {processed_frames_dir}")
