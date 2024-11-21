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

def load_cifar_data(dataset_name, seed=0):
    # Set the seed inside the function to ensure it's applied here as well
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the dataset with shuffled files
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Split validation set from training data
    val_size = int(0.1 * ds_info.splits['train'].num_examples)
    ds_val = ds_train.take(val_size)
    ds_train = ds_train.skip(val_size)
    
    # Dataset preparation with seeds in shuffle
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=45000, seed=seed, reshuffle_each_iteration=False)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    
    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(128)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(128)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    
    input_shape = (32, 32, 3)
    num_classes = 10 if dataset_name.lower() == 'cifar10' else 100
    
    return ds_train, ds_val, ds_test, input_shape, num_classes
