# This python script is to pre-process the image pair of blurry/sharp image
# The script will stitch two blurry and sharp image pair into one image
from PIL import Image
import os

# Define paths to folders containing images
root_folder = "./Gopro_align_data_raw/train"
# blurry_folder = "./Gopro_align_data/test/blur"
# sharp_folder = "./Gopro_align_data/test/sharp"
output_folder = "./Gopro_align_data/train"

subfolder = [d for d in os.listdir(root_folder)]

# Get a list of all image files in the folders
# blurry_images = [f for f in os.listdir(blurry_folder) if f.endswith('.png') or f.endswith('.tif')]
# sharp_images = [f for f in os.listdir(sharp_folder) if f.endswith('.png') or f.endswith('.tif')]

# Loop through each sample that containing blur/sharp image pairs
for subfolder_name in subfolder:
    blurry_images = [f for f in os.listdir(os.path.join(root_folder,subfolder_name,'blur')) if f.endswith('.png') or f.endswith('.tif')]
    sharp_images = [f for f in os.listdir(os.path.join(root_folder,subfolder_name,'sharp')) if f.endswith('.png') or f.endswith('.tif')]

    # Loop through each image in the folders and stitch them together horizontally
    for image_name in blurry_images:
        if image_name in sharp_images:
            # Open the blurry and sharp images using the Pillow library
            blurry_image = Image.open(os.path.join(root_folder,subfolder_name,'blur', image_name))
            sharp_image = Image.open(os.path.join(root_folder,subfolder_name,'sharp', image_name))
            
            # Get the width and height of the images
            width, height = blurry_image.size
            
            # Create a new blank image that's twice the width of the individual images
            new_image = Image.new('RGB', (width * 2, height))
            
            # Paste the blurry image on the left half of the new image
            new_image.paste(blurry_image, (0, 0))
            
            # Paste the sharp image on the right half of the new image
            new_image.paste(sharp_image, (width, 0))
            
            # Save the new image to the output folder
            new_image.save(os.path.join(output_folder, subfolder_name+image_name))
    print('Current subfolder process finished: %s' %(subfolder_name))
    