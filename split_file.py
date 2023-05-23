# This python script is to randomly split the images in parent_dir into train and val folder 

import os
import random
import shutil

# Set the percentage of images to use for validation
validation_percentage = 20

# Create subdirectories for train and validation data under parent_dir
parent_dir = '/mnt/f58069a5-1cf3-43b8-bb9b-ea74327327c9/WZH-DataCenter/PROCESS-SPT/2023/simPSF_results/MotionBlurDataset/Pixelized/20230517_PSNR22p61'
train_dir = 'train/'
val_dir = 'val/'
os.makedirs(os.path.join(parent_dir,train_dir), exist_ok=True)
os.makedirs(os.path.join(parent_dir,val_dir), exist_ok=True)

# Get all image files in the current directory
image_files = [f for f in os.listdir(parent_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

# Calculate the number of images to use for validation
num_validation_images = int(len(image_files) * validation_percentage / 100)

# Randomly select images for validation
validation_images = random.sample(image_files, num_validation_images)

# Move validation images to validation directory
for image_file in validation_images:
    shutil.move(os.path.join(parent_dir,image_file), os.path.join(parent_dir, val_dir, image_file))

# Move remaining images to train directory
for image_file in image_files:
    if image_file not in validation_images:
        shutil.move(os.path.join(parent_dir, image_file), os.path.join(parent_dir, train_dir, image_file))