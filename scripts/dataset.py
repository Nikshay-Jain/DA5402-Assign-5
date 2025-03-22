# Task 1

import torchvision.datasets as datasets
import os

# Define dataset storage path
DATASET_PATH = "data"
print("Downloading at:", DATASET_PATH)

# Download CIFAR-10 dataset
cifar10 = datasets.CIFAR10(root=DATASET_PATH, train=True, download=True)
classes = cifar10.classes     # CIFAR-10 class names

# Create subdirectories for each class
for class_name in classes:
    os.makedirs(os.path.join(DATASET_PATH, class_name), exist_ok=True)

# Save images into respective class folders
for idx, (image, label) in enumerate(cifar10):
    class_name = classes[label]
    image_path = os.path.join(DATASET_PATH, class_name, f"{idx}.png")
    image.save(image_path)

print("Dataset saved successfully in class-wise folders.")