from PIL import Image
import os

# input folder containing the images to rescale
input_folder = "/mnt/f58069a5-1cf3-43b8-bb9b-ea74327327c9/WZH-DataCenter/PROCESS-SPT/2023/simPSF_results/MotionBlurDataset/Raw/val"

# output folder to save the rescaled images
output_folder = "/mnt/f58069a5-1cf3-43b8-bb9b-ea74327327c9/WZH-DataCenter/PROCESS-SPT/2023/simPSF_results/MotionBlurDataset/Raw_rescaled/val"

# target width and height for rescaled images
target_width, target_height = (2560, 720)

# iterate through all the files in the input folder
for f in os.listdir(input_folder):
    # check if the file is an image
    if f.endswith('.jpg') or f.endswith('.png'):
        # open the image file using PIL
        image_path = os.path.join(input_folder, f)
        image = Image.open(image_path)
        
        # rescale the image to the target size using PIL's thumbnail function
        image.thumbnail((target_width, target_height))
        
        # save the rescaled image to the output folder with a new filename
        filename, extension = os.path.splitext(f)
        output_path = os.path.join(output_folder, f"{filename}_rescaled{extension}")
        image.save(output_path)
        
        # print the filename of the rescaled image
        print(f"Rescaled {f} and saved as {output_path}")