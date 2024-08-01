import os
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np




# Model class
class DeepLabv3Model(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3Model, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# Load the trained model
num_classes = 2  # Adjust based on your dataset
model = DeepLabv3Model(num_classes)
model.load_state_dict(
    torch.load('C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/weights/segmentation/DeepLabv3ONE/deeplabv3(2)_Far_Images.pth',
               map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load and preprocess the image
def load_image(image):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Process video
def process_video(input_video_path, output_video_path, window_size=(640, 480)):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmentation', window_size[0], window_size[1])

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Process every 10th frame
        if frame_counter % 10 == 0:
            # Process frame
            image = load_image(frame).to(device)
            with torch.no_grad():
                output = model(image)['out']
                output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Overlay and save the result
            original_size = frame.shape[:2][::-1]
            mask = (output * 255).astype(np.uint8)
            mask = cv2.resize(mask, original_size)  # Resize mask to original frame size
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            out.write(overlay)

            # Display the frame with overlay
            cv2.imshow('Segmentation', overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Write the original frame to maintain video length
            out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Example usage
input_video_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Video/vid1.mp4'  # Replace with the path to your input video
output_video_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Output_Video/DeepLab3.avi'  # Replace with the path to save the output video
process_video(input_video_path, output_video_path, window_size=(640, 480))
