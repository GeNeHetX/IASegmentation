import os
import random
import shutil
from tqdm import tqdm

# Define the paths and directories
image_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/image/'
label_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/labels/'

train_image_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/train'
validation_label_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/val_labels'
train_label_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/train_labels'
validation_image_folder = 'C:/Users/Hajar/Documents/dataset_2_classif/val'

validation_split = 0.20

# Create the train and validation folders if they don't exist
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(validation_image_folder, exist_ok=True)
os.makedirs(validation_label_folder, exist_ok=True)

# Get the list of image files
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Shuffle the image files randomly
random.shuffle(image_files)

# Define the split ratio between train and validation sets
validation_split = 0.20  # 20% for validation, 80% for training

# Calculate the number of images for validation set
num_validation_images = int(len(image_files) * validation_split)

# Move images and associated labels to the validation folder
for image_file in tqdm(image_files[:num_validation_images]):
    image_path = os.path.join(image_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + '.png'
    label_path = os.path.join(label_folder, label_file)
    shutil.move(image_path, validation_image_folder)
    shutil.move(label_path, validation_label_folder)

# Move images and associated labels to the train folder
for image_file in tqdm(image_files[num_validation_images:]):
    image_path = os.path.join(image_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + '.png'
    label_path = os.path.join(label_folder, label_file)
    shutil.move(image_path, train_image_folder)
    shutil.move(label_path, train_label_folder)

print("Images and labels successfully segregated into train and validation folders.")
