# scripts/evaluate_performance.py
import yaml
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Set random seed for reproducibility
random_seed = params["random_seed"]
tf.random.set_seed(random_seed)

# Define paths
test_dir = "data/test"

# Load the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode="int",
    seed=random_seed,
)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load the trained model
model = tf.keras.models.load_model("models/tuned_model")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = model.predict(test_dataset)
y_pred = np.argmax(y_pred, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_true, y_pred)
print("Classification Report:")
print(class_report)

# Save evaluation results to a file
os.makedirs("reports", exist_ok=True)
with open("reports/evaluation_report.txt", "w") as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")
    f.write("Classification Report:\n")
    f.write(class_report)

print("Evaluation complete. Results saved to 'reports/evaluation_report.txt'.")