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
image_transform_resized = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

mask_transform_resized = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

image_transform_standard = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

mask_transform_standard = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

# Create datasets and dataloaders
train_dataset_resized = SegmentationDataset(
    '/Training_Images/Only_Close_Images/images',
    '/Training_Images/Only_Close_Images/masks',
    transform=image_transform_resized, target_transform=mask_transform_resized)

train_dataset_standard = SegmentationDataset(
    '/Training_Images/Only_Far_Images/images',
    '/Training_Images/Only_Far_Images/masks',
    transform=image_transform_standard, target_transform=mask_transform_standard)


train_loader_resized = DataLoader(train_dataset_resized, batch_size=8, shuffle=True)
train_loader_standard = DataLoader(train_dataset_standard, batch_size=8, shuffle=True)


# Model class
class DeepLabv3Model(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3Model, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # Adding an extra convolutional layer to capture finer details
        self.extra_conv = nn.Conv2d(num_classes, 256, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)['out']
        x = self.extra_conv(x)
        x = self.relu(x)
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


# Custom loss function to penalize larger masks
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, masks):
        ce_loss = self.cross_entropy_loss(outputs, masks)

        # Calculate the number of foreground pixels
        foreground_pixels = torch.sum(masks == 1).float()

        # Penalize large masks
        penalty = foreground_pixels / (masks.size(0) * masks.size(1) * masks.size(2))

        # Combine the two losses
        loss = ce_loss + 0.1 * penalty

        return loss


criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 15

# Instantiate EMA
ema = EMA(model, decay=0.999)
ema.register()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Alternate between the two dataloaders
    for (images_resized, masks_resized), (images_standard, masks_standard) in zip(train_loader_resized, train_loader_standard):
        images_resized = images_resized.to(device).to(memory_format=torch.channels_last)
        masks_resized = masks_resized.to(device, dtype=torch.long)
        images_standard = images_standard.to(device).to(memory_format=torch.channels_last)
        masks_standard = masks_standard.to(device, dtype=torch.long)

        # Train on resized images
        optimizer.zero_grad()
        outputs_resized = model(images_resized).shape
        print(outputs_resized)
        loss_resized = criterion(outputs_resized, masks_resized)
        loss_resized.backward()
        optimizer.step()
        ema.update()
        running_loss += loss_resized.item()

        # Train on standard images
        optimizer.zero_grad()
        outputs_standard = model(images_standard)['out']
        loss_standard = criterion(outputs_standard, masks_standard)
        loss_standard.backward()
        optimizer.step()
        ema.update()
        running_loss += loss_standard.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(train_loader_resized) + len(train_loader_standard))}")

# Save the model
torch.save(model.state_dict(), '/weights/segmentation/DeepLabv3ONE/deeplabv3(3).pth')
