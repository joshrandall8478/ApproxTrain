# models/resnet.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers

def resnet_unit(inputs, filters, stride, shortcut_connection=True, weight_decay=1e-4, batch_norm_params=None):
    """Defines a single ResNet unit (block) without using class definitions."""
    if batch_norm_params is None:
        batch_norm_params = {
            'momentum': 0.99,
            'epsilon': 1e-3,
            'center': True,
            'scale': True,
        }
    kernel_regularizer = regularizers.l2(weight_decay)
    
    # First BatchNorm and ReLU
    x = layers.BatchNormalization(**batch_norm_params)(inputs)
    x = layers.ReLU()(x)
    
    # Shortcut connection
    shortcut = inputs
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same',
                                 use_bias=False, kernel_regularizer=kernel_regularizer)(x)
    
    # First Conv2D
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(x)
    
    # Second BatchNorm and ReLU
    x = layers.BatchNormalization(**batch_norm_params)(x)
    x = layers.ReLU()(x)
    
    # Second Conv2D
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(x)
    
    # Add shortcut connection
    if shortcut_connection:
        x = layers.Add()([x, shortcut])
    
    return x

def build_resnet(input_shape, num_classes, num_layers, weight_decay=1e-4):
    """Builds the ResNet model using the Keras Functional API."""
    if num_layers not in (20, 32, 44, 56, 110, 1202):
        raise ValueError('num_layers must be one of 20, 32, 44, 56, 110, or 1202.')
    
    num_units = (num_layers - 2) // 6
    batch_norm_params = {
        'momentum': 0.99,
        'epsilon': 1e-3,
        'center': True,
        'scale': True,
    }
    kernel_regularizer = regularizers.l2(weight_decay)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(inputs)
    
    # First block
    for _ in range(num_units):
        x = resnet_unit(x, filters=16, stride=1, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params)
    
    # Second block
    for i in range(num_units):
        stride = 2 if i == 0 else 1
        x = resnet_unit(x, filters=32, stride=stride, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params)
    
    # Third block
    for i in range(num_units):
        stride = 2 if i == 0 else 1
        x = resnet_unit(x, filters=64, stride=stride, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params)
    
    # Final BatchNorm and ReLU
    x = layers.BatchNormalization(**batch_norm_params)(x)
    x = layers.ReLU()(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=kernel_regularizer)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_resnet_cifar(input_shape, num_classes, depth):
    """Helper function to build ResNet models for CIFAR datasets."""
    return build_resnet(input_shape, num_classes, num_layers=depth, weight_decay=1e-4)
