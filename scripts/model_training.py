# scripts/model_training.py
import yaml
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Set random seed for reproducibility
random_seed = params["random_seed"]
tf.random.set_seed(random_seed)

# Define paths
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Load training and validation datasets using TensorFlow's image_dataset_from_directory
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(32, 32),  # Resize images to 32x32 (CIFAR-10 size)
    batch_size=params["model"]["batch_size"],
    label_mode="int",  # Labels are integers
    seed=random_seed,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(32, 32),
    batch_size=params["model"]["batch_size"],
    label_mode="int",
    seed=random_seed,
)

# Capture class names from the training dataset
class_names = train_data.class_names

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_dataset = train_data.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_data.map(lambda x, y: (normalization_layer(x), y))

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(params["model"]["num_filters"][0], (params["model"]["kernel_sizes"][0], params["model"]["kernel_sizes"][0]), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(params["model"]["num_filters"][1], (params["model"]["kernel_sizes"][1], params["model"]["kernel_sizes"][1]), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["model"]["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
epochs = params["model"]["epochs"]
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in tqdm(train_dataset):
        # Train on the batch
        model.train_on_batch(batch[0], batch[1])

    # Evaluate on the validation set
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Evaluate on the test set
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=params["model"]["batch_size"],
    label_mode="int",
    seed=random_seed,
    shuffle=False,  # Do not shuffle for evaluation
)

# Normalize test dataset
test_dataset = test_data.map(lambda x, y: (normalization_layer(x), y))

# Generate predictions
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = model.predict(test_dataset)
y_pred = np.argmax(y_pred, axis=1)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the trained model
model.save("models/tuned_model.keras")  # Save in Keras format
print("Model training complete. Model saved to 'models/tuned_model.keras'.")