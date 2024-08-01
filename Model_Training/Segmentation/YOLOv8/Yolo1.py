import os
from ultralytics import YOLO

# Load the model from a configuration file
model = YOLO("yolov8n-seg.pt")  # build a new model from scratch

# Define training parameters with data augmentation and learning rate adjustments
train_params = {
    'data': os.path.join("/Training_Images/Self_annotated/Health_Infected_Leaves", "config.yaml"),
    'epochs': 50,  # Increase epochs for better convergence
    'imgsz': 640,  # Image size
    'batch': 32,  # Adjust batch size according to GPU memory
    'rect': False,  # Rectangular training
    'device': '',  # Automatically select available CUDA device
    'workers': 8,  # Number of data loader workers
    'optimizer': 'Adam',  # Use Adam optimizer for better convergence
    'lr0': 0.001,  # Lower initial learning rate for fine-tuning
    'lrf': 0.01,  # Final learning rate (fraction of initial)
    'momentum': 0.9,  # Adjust momentum for Adam optimizer
    'weight_decay': 0.0005,  # Optimizer weight decay
    'warmup_epochs': 3.0,  # Warmup epochs
    'warmup_momentum': 0.8,  # Warmup initial momentum
    'warmup_bias_lr': 0.1,  # Warmup initial bias lr
    'box': 0.05,  # Box loss gain
    'degrees': 10,  # Image rotation degrees for more realistic augmentations
    'translate': 0.1,  # Image translation
    'scale': 0.5,  # Image scale
    'fliplr': 0.5,  # Image flip left-right
    'mosaic': 1.0,  # Mosaic augmentation
    'mixup': 0.2,  # Mixup augmentation
    'copy_paste': 0.5,  # Copy-paste augmentation
}

# Train the model with the specified parameters
results = model.train(**train_params)
