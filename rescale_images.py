input_folder = 'data'
output_folder = 'data-resized'

output_resolution = [224, 224]

from PIL import Image
import os
from resizeimage import resizeimage

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def rescale_images(input_folder, output_folder):
    class_folders = os.listdir(input_folder)
    for class_folder in class_folders:
        if not os.path.exists(os.path.join(output_folder, class_folder)):
            os.makedirs(os.path.join(output_folder, class_folder))
        for filename in os.listdir(os.path.join(input_folder, class_folder)):
            if filename.endswith('.jpg'):
                # print('Processing {}...'.format(filename))
                image = Image.open(os.path.join(input_folder, class_folder, filename))
                image = resizeimage.resize_contain(image, output_resolution)
                image = image.convert("RGB")
                print(os.path.join(output_folder, class_folder, filename))
                image.save(os.path.join(output_folder, class_folder, filename))

rescale_images(input_folder, output_folder)

