import tensorflow as tf
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

def build_lenet(model_name, input_shape, num_classes, lut_file, fpmode, AccumMode):
    if model_name == 'lenet300100':
        # LeNet-300-100 model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            denseam(300, activation='relu', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            denseam(100, activation='relu', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            denseam(num_classes, activation='softmax', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
        ])
    elif model_name == 'lenet5':
        # LeNet-5 model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            AMConv2D(6, kernel_size=5, padding='same', activation='tanh', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            tf.keras.layers.AveragePooling2D(),
            AMConv2D(16, kernel_size=5, padding='valid', activation='tanh', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            tf.keras.layers.AveragePooling2D(),
            AMConv2D(120, kernel_size=5, padding='valid', activation='tanh', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            tf.keras.layers.Flatten(),
            denseam(84, activation='tanh', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
            denseam(num_classes, activation='softmax', mant_mul_lut=lut_file, FPMode=fpmode, AccumMode=AccumMode),
        ])
    else:
        raise ValueError(f"Unsupported LeNet model: {model_name}")
    return model
