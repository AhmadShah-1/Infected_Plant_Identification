# This model adds transformations to further vary the input data by flipping, rotating, and chaning lighting to yield better accuracy in varying conditions

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Ensure mask values are within [0, 1]
        mask = np.array(mask)
        mask[mask > 0] = 1  # Assuming the mask is binary (0 for background, 1 for foreground)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

'''
    'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Far_Images/images',
    'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Far_Images/masks',
    
    'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Health_Infected_Leaves_With_Masks/images',
    'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Health_Infected_Leaves_With_Masks/masks',
    
    '''



# Create dataset and dataloader
train_dataset = SegmentationDataset(
    '/Training_Images/Field_batch_Images/images',
    '/Training_Images/Field_batch_Images/images',
    transform=image_transform, target_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


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


num_classes = 2  # Adjust based on your dataset
model = DeepLabv3Model(num_classes).to(memory_format=torch.channels_last)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).to(memory_format=torch.channels_last)


# Exponential Moving Average (EMA) class
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# Training setup
ema = EMA(model, decay=0.999)
ema.register()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        if images.size(0) == 1:  # Skip if batch size is 1
            print("batch skipped")
            continue

        images = images.to(device).to(memory_format=torch.channels_last)
        masks = masks.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)

        # Check output and mask dimensions
        print(f"Output size: {outputs.size()}, Mask size: {masks.size()}")

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        ema.update()

        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the model
torch.save(model.state_dict(), '/weights/segmentation/DeepLabv3ONE/deeplabv3(2)_Field_Images.pth')

