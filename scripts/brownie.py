import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define dataset paths
full_dataset_dir = "data/test"
v1_partition_dir = "partition/v1/test"

# Function to compute class distribution
def compute_class_distribution(dataset_path):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=(32, 32),
        batch_size=32,
        label_mode="int",
        shuffle=False,
    )
    class_names = dataset.class_names
    class_counts = {class_name: 0 for class_name in class_names}

    for _, y_batch in dataset:
        y_true_classes = y_batch.numpy()
        for class_idx in y_true_classes:
            class_counts[class_names[class_idx]] += 1

    return class_counts

# Compute class distributions
full_dataset_counts = compute_class_distribution(full_dataset_dir)
v1_counts = compute_class_distribution(v1_partition_dir)

# Print class distributions
print("\nClass Distribution in Full Test Dataset:")
for class_name, count in full_dataset_counts.items():
    print(f"{class_name}: {count}")

print("\nClass Distribution in V1 Partition Test Dataset:")
for class_name, count in v1_counts.items():
    print(f"{class_name}: {count}")

# Save to evaluation report
evaluation_report = {}  # Initialize evaluation_report as a dictionary
evaluation_report["class_distribution"] = {
    "full_dataset": full_dataset_counts,
    "v1_partition": v1_counts,
}

# Plot class distribution
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.4  # Set bar width for better comparison
x_labels = list(full_dataset_counts.keys())

# Bar positions
x = np.arange(len(x_labels))
ax.bar(x - bar_width/2, full_dataset_counts.values(), bar_width, label="Full Dataset", color="skyblue")
ax.bar(x + bar_width/2, v1_counts.values(), bar_width, label="V1 Partition", color="orange")

# Labels and title
ax.set_xlabel("Class")
ax.set_ylabel("Number of Samples")
ax.set_title("Class Distribution in Full Test Dataset vs. V1 Test Partition")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45)
ax.legend()

plt.savefig("figures/class_distribution.png")