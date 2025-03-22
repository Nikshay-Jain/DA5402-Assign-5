# scripts/evaluate_performance.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Set random seed for reproducibility
random_seed = params["random_seed"]
tf.random.set_seed(random_seed)

# Define paths
test_dir = "data/test"
model_path = "models/tuned_model.keras"

# Load the test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),  # Resize images to 32x32 (CIFAR-10 size)
    batch_size=params["model"]["batch_size"],
    label_mode="int",  # Labels are integers
    seed=random_seed,
    shuffle=False,  # Do not shuffle for evaluation
)

# Capture class names from the test dataset
class_names = test_data.class_names

# Normalize pixel values to [0, 1] and convert labels to one-hot encoding
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
test_dataset = test_data.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=10)))

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Generate predictions
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = model.predict(test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Compute overall accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Overall Accuracy: {accuracy:.4f}")

# Generate a classification report for detailed metrics per class
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(12, 9))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Class-wise accuracy
class_accuracy = {}
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_true_classes == i)[0]
    correct_predictions = np.sum(y_pred_classes[class_indices] == y_true_classes[class_indices])
    total_samples = len(class_indices)
    class_accuracy[class_name] = correct_predictions / total_samples
    print(f"Accuracy for class {class_name}: {class_accuracy[class_name]:.4f}")

# Misclassification table
misclassification_table = {}
for i, true_class in enumerate(class_names):
    misclassification_table[true_class] = {}
    for j, pred_class in enumerate(class_names):
        if i != j:
            misclassification_table[true_class][pred_class] = np.sum((y_true_classes == i) & (y_pred_classes == j))

# Print misclassification table
print("\nMisclassification Table:")
for true_class in class_names:
    print(f"\nTrue Class: {true_class}")
    for pred_class in class_names:
        if true_class != pred_class:
            print(f"Misclassified as {pred_class}: {misclassification_table[true_class][pred_class]}")

# Save evaluation report
evaluation_report = {
    "overall_accuracy": accuracy,
    "class_accuracy": class_accuracy,
    "misclassification_table": misclassification_table,
}

# Save the evaluation report to a file
import json
with open("reports/evaluation_report.json", "w") as f:
    json.dump(evaluation_report, f, indent=4)

print("Evaluation complete. Report saved to 'reports/evaluation_report.json'.")