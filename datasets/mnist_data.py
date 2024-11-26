import tensorflow as tf
import tensorflow_datasets as tfds
import random
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
def set_seed(seed=0):
    """
    Set seeds for reproducibility across various libraries and TensorFlow configurations.
    """
    # Set PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set Python's random module seed
    random.seed(seed)
    
    # Set NumPy's RNG seed
    np.random.seed(seed)
    
    # Set TensorFlow's RNG seed
    tf.random.set_seed(seed)
    
    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Optional: Enable if using cuDNN

def load_mnist_data(seed=0):
    # Set seeds for reproducibility
    set_seed(seed)
    batch_size = 128
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape data to add channel dimension (28, 28, 1)
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Split training data into training and validation sets (80-20 split)
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=0.2,
        random_state=seed,
        stratify=train_labels,
        shuffle=True
    )

    # Number of samples
    num_train_samples = train_images.shape[0]
    num_val_samples = val_images.shape[0]
    num_test_samples = test_images.shape[0]

    # Define input shape and number of classes
    input_shape = train_images.shape[1:]
    num_classes = 10

     # Compute per-channel mean and std for normalization
    mean = np.mean(train_images, axis=(0, 1, 2))
    std = np.std(train_images, axis=(0, 1, 2))
    
    # Define the preprocessing function
    def preprocess_input(x):
        """Preprocess input by normalizing with mean and std."""
        x = (x - mean) / (std + 1e-7)
        return x
    
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.125,  # 0.125 * 28 ≈ 3.5 pixels
        height_shift_range=0.125,
        rotation_range=10,        # Rotate images by up to 10 degrees
        zoom_range=0.1,            # Zoom in by up to 10%
        horizontal_flip=False,    # MNIST digits are symmetric; flipping isn't necessary
        fill_mode='reflect'       # Fill missing pixels after transformations
    )
    
    # Only normalization for validation data
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Only normalization for test data
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow(
        train_images, 
        train_labels,
        batch_size=batch_size,
        shuffle=True
    )
    val_generator = val_datagen.flow(
        val_images, 
        val_labels,
        batch_size=batch_size,
        shuffle=False
    )
    test_generator = test_datagen.flow(
        test_images, 
        test_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, input_shape, num_classes