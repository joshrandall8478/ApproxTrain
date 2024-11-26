# datasets/cifar_data.py

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split



def load_cifar_data(dataset_name, batch_size=128):
    """Loads CIFAR data and returns training and validation generators."""
    if dataset_name == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        num_classes = 10
    elif dataset_name == 'cifar100':
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Compute per-channel mean and std for normalization
    mean = np.mean(train_images, axis=(0, 1, 2))
    std = np.std(train_images, axis=(0, 1, 2))

    # Define the split ratio
    train_ratio = 0.9  # 90% for training
    test_ratio = 0.1   # 10% for validation

    # Perform the split
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=test_ratio,
        train_size=train_ratio,
        random_state=42,    # Ensures reproducibility
        shuffle=True,       # Shuffles the data before splitting
        stratify=train_labels          # Maintains class distribution
    )

    train_images = train_images.astype('float32')
    val_images = val_images.astype('float32')
    test_images = test_images.astype('float32')
    
    
    
    def preprocess_input(x):
        """Preprocess input by normalizing with mean and std."""
        x = (x - mean) / (std + 1e-7)
        # pad width with 4 pixels, then randomly crop to 32x32
        # x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        # x = tf.image.random_crop(x, size=[32, 32, 3])
        return x
    
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.125,  # 0.125 * 32 = 4 pixels
        height_shift_range=0.125,
        horizontal_flip=True,
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
    train_generator = train_datagen.flow(train_images, tf.keras.utils.to_categorical(train_labels, num_classes),
                                         batch_size=batch_size)
    val_generator = val_datagen.flow(val_images, tf.keras.utils.to_categorical(val_labels, num_classes),
                                     batch_size=batch_size)
    test_generator = test_datagen.flow(test_images, tf.keras.utils.to_categorical(test_labels, num_classes),
                                       batch_size=batch_size)
    
    input_shape = train_images.shape[1:]
    return train_generator, val_generator, test_generator, input_shape, num_classes



