# scripts/model_training.py
import os, yaml
import keras_tuner as kt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
num_classes = 10

# Normalize pixel values to [0, 1] and convert labels to one-hot encoding
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_dataset = train_data.map(lambda x, y: (normalization_layer(x), to_categorical(y, num_classes=10)))
val_dataset = val_data.map(lambda x, y: (normalization_layer(x), to_categorical(y, num_classes=10)))

# Cache and prefetch datasets for efficient loading
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define a function to build the model for hyperparameter tuning
def build_model(hp):
    model = models.Sequential()
    model.add(Input(shape=(32, 32, 3)))

    # First Convolutional Layer
    model.add(layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        padding='same',
        activation='relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        padding='same',
        activation='relu'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Dense Layer
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp.Float('l2_reg', min_value=0.001, max_value=0.01, step=0.001)))
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Initialize the Keras Tuner (RandomSearch)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,                       # Number of hyperparameter combinations to try
    executions_per_trial=1,              # Number of models to train per trial
    directory='hyperparameter_tuning'
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Perform hyperparameter tuning
tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=params["model"]["epochs"],
    callbacks=[early_stopping]
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model with the full training dataset
history = best_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=params["model"]["epochs"],
    callbacks=[early_stopping]
)

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the best model
best_model.save("models/tuned_model.keras")
print("Model training complete. Best model saved to 'models/tuned_model.keras'.")