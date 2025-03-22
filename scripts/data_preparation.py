# scripts/data_preparation.py
import yaml
import os
import random
import shutil

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

random_seed = params["random_seed"]
random.seed(random_seed)

# Define paths
raw_data_dir = "data"  # Folder containing class-wise raw images
input_dir = "data/20k_partition"  # Folder to store renamed images
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Create output directories
os.makedirs(input_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to copy and rename images to include class name
def copy_and_rename_images(raw_data_dir, input_dir):
    for class_name in os.listdir(raw_data_dir):
        class_dir = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            print(f"Processing class: {class_name}")
            for image_name in os.listdir(class_dir):
                if image_name.endswith(('.png')):  # Filter allowed formats
                    old_path = os.path.join(class_dir, image_name)
                    new_name = f"{class_name}_{image_name}"  # Add class name to the filename
                    new_path = os.path.join(input_dir, new_name)
                    shutil.copy(old_path, new_path)
                    print(f"Copied and renamed {old_path} to {new_path}")

# Copy and rename images from raw data directory to input directory
copy_and_rename_images(raw_data_dir, input_dir)

# Get list of all images
all_images = []
for image_name in os.listdir(input_dir):
    if image_name.endswith(('.png')):  # Filter allowed formats
        image_path = os.path.join(input_dir, image_name)
        all_images.append(image_path)
        print(f"Added image: {image_path}")

# Shuffle and split the data
random.shuffle(all_images)
train_size = int(len(all_images) * params["data"]["train_ratio"])
val_size = int(len(all_images) * params["data"]["val_ratio"])

train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Function to copy images to respective folders
def copy_images(images, dest_dir):
    for image in images:
        # Extract class name from the filename (e.g., "airplane_001.png" -> "airplane")
        class_name = os.path.basename(image).split("_")[0]
        dest_class_dir = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        shutil.copy(image, dest_class_dir)
        print(f"Copied {image} to {dest_class_dir}")

# Copy images to respective folders
copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)

print("Data preparation complete.")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")