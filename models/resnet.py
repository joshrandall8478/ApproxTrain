# models/resnet.py

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

def resnet_unit(inputs, filters, stride, lut_file, FPMode, shortcut_connection=True, weight_decay=1e-4, batch_norm_params=None, tensorflow_original=False):

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
    x = BatchNormalization(**batch_norm_params)(inputs)
    x = ReLU()(x)
    
    # Shortcut connection
    shortcut = inputs
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same',
                                 use_bias=False, kernel_regularizer=kernel_regularizer)(x) \
                if tensorflow_original else \
                AMConv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias= False, 
                         mant_mul_lut=lut_file, FPMode=FPMode)(x)
    
    # First Conv2D
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(x) \
        if tensorflow_original else \
        AMConv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias= False,\
                 kernel_regularizer=kernel_regularizer, mant_mul_lut=lut_file, FPMode=FPMode)(x)
    
    # Second BatchNorm and ReLU
    x = BatchNormalization(**batch_norm_params)(x)
    x = ReLU()(x)
    
    # Second Conv2D
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(x)\
        if tensorflow_original else \
        AMConv2D(filters, kernel_size=3, strides=1, padding='same', use_bias= False,\
                    kernel_regularizer=kernel_regularizer, mant_mul_lut=lut_file, FPMode=FPMode)(x)
    
    # Add shortcut connection
    if shortcut_connection:
        x = Add()([x, shortcut])
    
    return x

def build_resnet(input_shape, num_classes, num_layers, lut_file, FPMode, weight_decay=1e-4,tensorflow_original=False):
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
    x = Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_regularizer=kernel_regularizer)(inputs)\
        if tensorflow_original else \
        AMConv2D(16, kernel_size=3, strides=1, padding='same', use_bias= False,\
                 kernel_regularizer=kernel_regularizer, mant_mul_lut=lut_file, FPMode=FPMode)(inputs)
    
    # First block
    for _ in range(num_units):
        x = resnet_unit(x, filters=16, stride=1, lut_file=lut_file, FPMode=FPMode, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params, tensorflow_original=tensorflow_original)
    
    # Second block
    for i in range(num_units):
        stride = 2 if i == 0 else 1
        x = resnet_unit(x, filters=32, stride=stride, lut_file=lut_file, FPMode=FPMode, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params, tensorflow_original=tensorflow_original)
    
    # Third block
    for i in range(num_units):
        stride = 2 if i == 0 else 1
        x = resnet_unit(x, filters=64, stride=stride, lut_file=lut_file, FPMode=FPMode, shortcut_connection=True,
                        weight_decay=weight_decay, batch_norm_params=batch_norm_params, tensorflow_original=tensorflow_original)
    
    # Final BatchNorm and ReLU
    x = BatchNormalization(**batch_norm_params)(x)
    x = ReLU()(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax',
                           kernel_regularizer=kernel_regularizer)(x) \
        if tensorflow_original else \
        denseam(num_classes, activation='softmax', kernel_regularizer=kernel_regularizer,\
                mant_mul_lut=lut_file, FPMode=FPMode)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def build_resnet_cifar(input_shape, num_classes, depth, tensorflow_original=False):
    """Helper function to build ResNet models for CIFAR datasets."""
    return build_resnet(input_shape, num_classes, num_layers=depth, weight_decay=1e-4, tensorflow_original=tensorflow_original)
