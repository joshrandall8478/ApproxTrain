import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train.squeeze(), num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test.squeeze(), num_classes=10)


lut_file = "lut/MBM_7.bin"

from tensorflow.keras.initializers import HeNormal
# Residual Block with Approximate Conv2D
# def residual_block(inputs, filters, stride=1, downsample=False):
    
#     x = AMConv2D(filters, kernel_size=3, strides=stride, padding='same', activation='relu', mant_mul_lut=lut_file)(inputs)
#     x = AMConv2D(filters, kernel_size=3, padding='same', activation=None, mant_mul_lut=lut_file)(x)
    
#     shortcut = inputs
#     if downsample:
#         shortcut = AMConv2D(filters, kernel_size=1, strides=stride, padding='same', activation=None, mant_mul_lut=lut_file)(inputs)

#     x = tf.keras.layers.Add()([x, shortcut])
#     x = tf.keras.layers.ReLU()(x)
#     return x
def residual_block(inputs, filters, stride=1, downsample=False):
    x = AMConv2D(filters, kernel_size=3, strides=stride, padding='same', activation='relu', 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)
    x = AMConv2D(filters, kernel_size=3, padding='same', activation=None, 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(x)
    
    shortcut = inputs
    if downsample:
        shortcut = AMConv2D(filters, kernel_size=1, strides=stride, padding='same', activation=None, 
                            kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# from tensorflow.keras.layers import Conv2D, Dense

# def residual_block(inputs, filters, stride=1, downsample=False):
#     x = Conv2D(filters, kernel_size=3, strides=stride, padding='same', activation='relu')(inputs)
#     x = Conv2D(filters, kernel_size=3, padding='same', activation=None)(x)

#     shortcut = inputs
#     if downsample:
#         shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', activation=None)(inputs)

#     x = tf.keras.layers.Add()([x, shortcut])
#     x = tf.keras.layers.ReLU()(x)
#     return x

# Build ResNet-like Model
def resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = AMConv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', mant_mul_lut=lut_file)(inputs)

    # Residual Blocks
    x = residual_block(x, 64)

    # x = residual_block(x, 128, stride=2, downsample=True)
    # x = residual_block(x, 128)

    # # x = residual_block(x, 256, stride=2, downsample=True)
    # # x = residual_block(x, 256)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = denseam(num_classes, activation='softmax', mant_mul_lut=lut_file)(x)

    return tf.keras.Model(inputs, outputs)


# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Create Adam optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Compile and Train
model = resnet()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)