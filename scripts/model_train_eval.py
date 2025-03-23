# Task 3 and 4

import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os, yaml, logging, subprocess, json

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configure logging
log_dir = "logs"                      # Directory to save log files
os.makedirs(log_dir, exist_ok=True)   # Create logs directory if it doesn't exist

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"model_training_{datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}.log")),  # Save logs to a file
        logging.StreamHandler()       # Print logs to the console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Model training script started.")

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Set random seed for reproducibility
random_seed = params["random_seed"]
tf.random.set_seed(random_seed)

# Define dataset partitions to process
task3_dataset = "data"
partitions = ["v1", "v2", "v3", "v1+v2", "v1+v2+v3"]

# Function to train and evaluate a model for a given partition
def train_and_evaluate(partition, random_seed):
    logger.info(f"Processing partition: {partition}")

    # Define paths
    data_dir = f"partition/{partition}"                                             # Directory for the current partition
    model_path = f"models/tuned_model_{partition}_seed{random_seed}.keras"          # Save model with partition and seed
    report_path = f"reports/evaluation_report_{partition}_seed{random_seed}.json"   # Save evaluation report

    if partition=="data":
        data_dir = "data"
    else:
        # Pull dataset from DVC
        logger.info(f"Pulling dataset {partition} from DVC...")
        subprocess.run(["dvc", "pull", data_dir], check=True)

    # Load training, validation, and test datasets using TensorFlow's image_dataset_from_directory
    logger.info("Loading datasets...")
    
    train_data = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(32, 32),  # Resize images to 32x32 (CIFAR-10 size)
        batch_size=params["model"]["batch_size"],
        label_mode="int",  # Labels are integers
        seed=random_seed,
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=(32, 32),
        batch_size=params["model"]["batch_size"],
        label_mode="int",
        seed=random_seed,
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=(32, 32),
        batch_size=params["model"]["batch_size"],
        label_mode="int",
        seed=random_seed,
        shuffle=False,  # Do not shuffle for evaluation
    )

    # Capture class names from the training dataset
    class_names = train_data.class_names
    num_classes = 10

    # Normalize pixel values to [0, 1] and convert labels to one-hot encoding
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_data.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=10)))
    val_dataset = val_data.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=10)))
    test_dataset = test_data.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=10)))

    # Cache and prefetch datasets for efficient loading
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4])),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Initialize the Keras Tuner (RandomSearch)
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=2,                   # Number of hyperparameter combinations to try
        executions_per_trial=1,         # Number of models to train per trial
        directory='hyperparameter_tuning',
        project_name=f'cifar10_tuning_{partition}_seed{random_seed}'
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Perform hyperparameter tuning
    logger.info("Starting hyperparameter tuning...")
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=params["model"]["epochs"],
        callbacks=[early_stopping]
    )

    # Get the best model from tuning
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model with the full training dataset
    logger.info("Training the best model...")
    history = best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params["model"]["epochs"],
        callbacks=[early_stopping]
    )

    # Save the best model
    best_model.save(model_path)
    logger.info(f"Model saved to '{model_path}'.")

    # Evaluate the best model on the test dataset
    logger.info("Evaluating the best model on the test dataset...")
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred = best_model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Generate classification report
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
    logger.info("Classification Report:")
    logger.info(class_report)

    # Generate confusion matrix
    confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
    logger.info("Confusion Matrix:")
    logger.info(confusion_mtx)

    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Plot the confusion matrix
    plt.figure(figsize=(12, 9))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {partition}_seed{random_seed}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"figures/confusion_matrix_{partition}_seed{random_seed}'.png")

    # Class-wise accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true_classes == i)[0]
        correct_predictions = np.sum(y_pred_classes[class_indices] == y_true_classes[class_indices])
        total_samples = len(class_indices)
        class_accuracy[class_name] = float(correct_predictions / total_samples)  # Convert to float
        print(f"Accuracy for class {class_name}: {class_accuracy[class_name]:.4f}")

    # Misclassification table
    misclassification_table = {}
    for i, true_class in enumerate(class_names):
        misclassification_table[true_class] = {}
        for j, pred_class in enumerate(class_names):
            if i != j:
                misclassification_table[true_class][pred_class] = int(np.sum((y_true_classes == i) & (y_pred_classes == j)))  # Convert to int

    # Print misclassification table
    print("\nMisclassification Table:")
    for true_class in class_names:
        print(f"\nTrue Class: {true_class}")
        for pred_class in class_names:
            if true_class != pred_class:
                print(f"Misclassified as {pred_class}: {misclassification_table[true_class][pred_class]}")

    # Save evaluation report
    evaluation_report = {
        "partition": partition,
        "random_seed": random_seed,
        "overall_accuracy": float(accuracy_score(y_true_classes, y_pred_classes)),
        "class_accuracy": {class_name: float(np.mean(y_pred_classes[y_true_classes == i] == i)) for i, class_name in enumerate(class_names)},
        "classification_report": class_report,
        "misclassification_table": misclassification_table,
    }

    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)
    logger.info(f"Evaluation report saved to '{report_path}'.")

# task 3 processing
train_and_evaluate(task3_dataset, random_seed)

# Process each partition for task 4
for partition in partitions:
    train_and_evaluate(partition, random_seed)

logger.info("All partitions processed. Model training complete.")