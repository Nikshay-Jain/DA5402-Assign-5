import os
import random
import shutil

# Define paths
PARTITION_DIR = "partition"
DATASET_PATH = "data"
os.makedirs(PARTITION_DIR, exist_ok=True)

# Define partition names
partitions = ["v1", "v2", "v3"]

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

    # Function to copy images to partitions
    def copy_images(image_list, partition):
        for img in image_list:
            src = os.path.join(class_dir, img)
            dst = os.path.join(PARTITION_DIR, partition, class_name, img)
            shutil.copy(src, dst)

    # Copy images to respective partitions
    copy_images(v1, "v1")
    copy_images(v2, "v2")
    copy_images(v3, "v3")

print("Class-wise partitions created successfully!")