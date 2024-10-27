import os
import requests
from pycocotools.coco import COCO

# Define COCO dataset paths
coco_annotation_file = 'annotations/instances_train2017.json'

# Load COCO annotations
coco = COCO(coco_annotation_file)

# Specify the class name (e.g., 'person')
class_name = 'couch'

# Get the category ID for the desired class
cat_ids = coco.getCatIds(catNms=[class_name])

# Get all image IDs for the desired class
img_ids = coco.getImgIds(catIds=cat_ids)

# Specify the number of samples you want to download
num_samples = 50

# Download the images
output_folder = 'images/couch'
os.makedirs(output_folder, exist_ok=True)

for i, img_id in enumerate(img_ids[:num_samples]):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_data = requests.get(img_url).content

    # Save the image to the output folder
    img_filename = os.path.join(output_folder, img_info['file_name'])
    with open(img_filename, 'wb') as handler:
        handler.write(img_data)

    print(f"Downloaded {i + 1}/{num_samples}: {img_info['file_name']}")

print(f"Successfully downloaded {num_samples} images for class '{class_name}'.")