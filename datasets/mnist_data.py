import tensorflow as tf
import tensorflow_datasets as tfds
import random
import numpy as np
import os
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
    # Set the seed inside the function to ensure it's applied here as well
    set_seed(seed)
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[:80%]', 'train[80%:]', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Dataset preparation 
    ds_train = ds_train.map(normalize_img).cache().shuffle(48000, seed=seed, reshuffle_each_iteration=False).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    input_shape = (28, 28, 1)
    num_classes = 10

    return ds_train, ds_val, ds_test, input_shape, num_classes




