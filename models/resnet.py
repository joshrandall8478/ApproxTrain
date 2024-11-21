import tensorflow as tf
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

def build_resnet(model_name, input_shape, num_classes, lut_file, fpmode):
    if model_name == 'resnet18':
        return ResNet(input_shape, num_classes, [2, 2, 2, 2], lut_file, fpmode)
    elif model_name == 'resnet34':
        return ResNet(input_shape, num_classes, [3, 4, 6, 3], lut_file, fpmode)
    elif model_name == 'resnet50':
        return ResNet50(input_shape, num_classes, lut_file, fpmode)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

def ResNet(input_shape, num_classes, layers_per_block, lut_file, fpmode):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = AMConv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, mant_mul_lut=lut_file, FPMode=fpmode)(inputs)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    filters = [64, 128, 256, 512]
    for i, num_blocks in enumerate(layers_per_block):
        for j in range(num_blocks):
            strides = 1
            if i != 0 and j == 0:
                strides = 2
            x = basic_block(x, filters[i], strides, lut_file, fpmode)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = denseam(num_classes, activation='softmax', mant_mul_lut=lut_file, FPMode=fpmode)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def basic_block(input_tensor, filters, strides, lut_file, fpmode):
    x = AMConv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False, mant_mul_lut=lut_file, FPMode=fpmode)(input_tensor)
    x = tf.keras.layers.BatchNormalization()
    x = tf.keras.layers.ReLU()(x)

    x = AMConv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, mant_mul_lut=lut_file, FPMode=fpmode)(x)
    x = tf.keras.layers.BatchNormalization()

    shortcut = input_tensor
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = AMConv2D(filters, kernel_size=1, strides=strides, use_bias=False, mant_mul_lut=lut_file, FPMode=fpmode)(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization()

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def ResNet50(input_shape, num_classes, lut_file, fpmode):
    # Simplified implementation for ResNet-50
    # You may need to adjust this for a complete ResNet-50 model
    from tensorflow.keras.applications import ResNet50
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)

    # Replace Conv2D layers with AMConv2D
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            config = layer.get_config()
            new_layer = AMConv2D.from_config(config, mant_mul_lut=lut_file, FPMode=fpmode)
            base_model._layers[base_model.layers.index(layer)] = new_layer

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = denseam(num_classes, activation='softmax', mant_mul_lut=lut_file, FPMode=fpmode)(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model
