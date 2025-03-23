# Task 2

import os, yaml
import random
import shutil

# Define paths
PARTITION_DIR = "partition"
DATASET_PATH = "data"
os.makedirs(PARTITION_DIR, exist_ok=True)

# Define partition names
partitions = ["v1", "v2", "v3"]

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Define split ratios for train, val, test
train_ratio = params["partition"]["train_ratio"]
val_ratio = params["partition"]["val_ratio"]
test_ratio = params["partition"]["test_ratio"]

# List of 10 classes to keep
classes_to_keep = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Create class-wise folders inside each partition
for partition in partitions:
    for class_name in os.listdir(DATASET_PATH):
        class_dir = os.path.join(DATASET_PATH, class_name)
        # Ignore directories that contain subdirectories
        if os.path.isdir(class_dir) and not any(os.path.isdir(os.path.join(class_dir, sub)) for sub in os.listdir(class_dir)):
            os.makedirs(os.path.join(PARTITION_DIR, partition, class_name), exist_ok=True)

# Process each valid class
for class_name in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, class_name)
    
    # Ignore directories that contain subdirectories
    if not os.path.isdir(class_dir) or any(os.path.isdir(os.path.join(class_dir, sub)) for sub in os.listdir(class_dir)):
        continue

    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Shuffle images within the class

    # Split images equally into v1, v2, v3
    split1 = len(images) // 3
    split2 = 2 * (len(images) // 3)

    v1, v2, v3 = images[:split1], images[split1:split2], images[split2:]

    # Function to copy images to partitions with train/val/test splits
    def copy_images_with_splits(image_list, partition):
        # Create train, val, test directories
        train_dir = os.path.join(PARTITION_DIR, partition, "train", class_name)
        val_dir = os.path.join(PARTITION_DIR, partition, "val", class_name)
        test_dir = os.path.join(PARTITION_DIR, partition, "test", class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Split images into train, val, test
        train_split = int(len(image_list) * train_ratio)
        val_split = int(len(image_list) * (train_ratio + val_ratio))
        train_images = image_list[:train_split]
        val_images = image_list[train_split:val_split]
        test_images = image_list[val_split:]

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, img))

    # Copy images to respective partitions with train/val/test splits
    copy_images_with_splits(v1, "v1")
    copy_images_with_splits(v2, "v2")
    copy_images_with_splits(v3, "v3")

    # Create v1+v2 partition with train/val/test splits
    os.makedirs(os.path.join(PARTITION_DIR, "v1+v2", class_name), exist_ok=True)
    copy_images_with_splits(v1 + v2, "v1+v2")

    # Create v1+v2+v3 partition with train/val/test splits
    os.makedirs(os.path.join(PARTITION_DIR, "v1+v2+v3", class_name), exist_ok=True)
    copy_images_with_splits(v1 + v2 + v3, "v1+v2+v3")

# Cleanup: Delete all folders except train, val, test in each partition
for partition in partitions + ["v1+v2", "v1+v2+v3"]:
    partition_dir = os.path.join(PARTITION_DIR, partition)
    if os.path.exists(partition_dir):
        for item in os.listdir(partition_dir):
            item_path = os.path.join(partition_dir, item)
            if os.path.isdir(item_path) and item not in ["train", "val", "test"]:
                shutil.rmtree(item_path)

# Cleanup: Delete all folders except the 10 classes in train, val, test directories
for partition in partitions + ["v1+v2", "v1+v2+v3"]:
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(PARTITION_DIR, partition, split)
        if os.path.exists(split_dir):
            for item in os.listdir(split_dir):
                item_path = os.path.join(split_dir, item)
                if os.path.isdir(item_path) and item not in classes_to_keep:
                    shutil.rmtree(item_path)

print("Class-wise partitions with train/val/test splits created and cleaned up successfully!")