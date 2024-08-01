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
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


        # Adding extra convolutional layers to capture finer details
        self.extra_conv1 = nn.Conv2d(num_classes, 256, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()

        self.extra_conv2 = nn.Conv2d(256, 256, kernel_size=(5, 5), padding=2)  # Adjust padding to keep dimensions
        self.relu2 = nn.ReLU()



    def forward(self, x):
        x = self.model(x)['out']
        x = self.extra_conv1(x)
        x = self.relu1(x)
        x = self.extra_conv2(x)
        x = self.relu2(x)
        return x


# Load the trained model
num_classes = 2  # Adjust based on your dataset
model = DeepLabv3Model(num_classes)
model.load_state_dict(
    torch.load('/weights/segmentation/DeepLabv3ONE/deeplabv3(2)_Field_Images.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Process and save all images in a directory
def process_images_in_directory(directory_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = load_image(image_path).to(device)

            with torch.no_grad():
                output = model(image)
                output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            overlay_and_save_result(image_path, output, output_directory, filename)


# Overlay and save the result
def overlay_and_save_result(image_path, output, output_directory, filename):
    original_image = cv2.imread(image_path)
    original_size = original_image.shape[:2][::-1]
    mask = (output * 255).astype(np.uint8)
    mask = cv2.resize(mask, original_size)  # Resize mask to original image size
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Blend the original image and the mask
    overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)

    cv2.imwrite(os.path.join(output_directory, f"overlay_{filename}"), overlay)


# Example usage
# directory_path = '../Training_Images/Health_Infected_Leaves_With_Masks/images'  # Replace with the path to your directory
# directory_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Random/images'  # Replace with the path to your directory
directory_path = '/Training_Images/Field_batch_Images/images'
output_directory = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Random/output'  # Replace with the path to save outputs
process_images_in_directory(directory_path, output_directory)
