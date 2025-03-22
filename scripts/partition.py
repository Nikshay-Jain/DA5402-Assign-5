import os, random, shutil

# Define partition directories
PARTITION_DIR = "partition"
DATASET_PATH = "data"
os.makedirs(PARTITION_DIR, exist_ok=True)

# Get all images
all_images = []
for class_name in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_dir):
        for img in os.listdir(class_dir):
            all_images.append((class_name, img))

# Shuffle & split into three partitions
random.shuffle(all_images)
v1, v2, v3 = all_images[:20000], all_images[20000:40000], all_images[40000:]

# Function to save partition
def save_partition(partition, name):
    partition_path = os.path.join(PARTITION_DIR, name)
    os.makedirs(partition_path, exist_ok=True)
    print("Creating partitions...", name)
    for class_name, img in partition:
        src = os.path.join(DATASET_PATH, class_name, img)
        dst = os.path.join(partition_path, img)
        shutil.copy(src, dst)

save_partition(v1, "v1")
save_partition(v2, "v2")
save_partition(v3, "v3")
print("Partitions created successfully.")