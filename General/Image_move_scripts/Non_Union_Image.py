# This file compares files from folder 1 to folder 2, and any images not present in folder 2 are saved to folder 3 (it moves them not copies so note that)


import os
import shutil

# Define the paths to the folders
'''
C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Health_Infected_Leaves_With_Masks/masks
C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Far_Images/masks
C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Close_Images/masks
'''


folder1_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Health_Infected_Leaves_With_Masks/masks'
folder2_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Far_Images/masks'
folder3_path = 'C:/Users/alex1/Desktop/Ahmad_Stuff/Code/Project1/Training_Images/Only_Close_Images/masks'

# Create folder 3 if it doesn't exist
if not os.path.exists(folder3_path):
    os.makedirs(folder3_path)

# Get the list of images in folder 1 and folder 2
images_folder1 = set(os.listdir(folder1_path))
images_folder2 = set(os.listdir(folder2_path))

# Identify images that are in folder 1 but not in folder 2
unique_images = images_folder1 - images_folder2

# Move the identified images to folder 3
for image in unique_images:
    source = os.path.join(folder1_path, image)
    destination = os.path.join(folder3_path, image)
    shutil.move(source, destination)

print(f"Moved {len(unique_images)} images to {folder3_path}")
