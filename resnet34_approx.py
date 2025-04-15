import tensorflow as tf
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Redefined identity and convolutional blocks (same custom layers)
def identity_block(x, filters, lut_file):
    x_skip = x
    x = AMConv2D(filters=filters, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = AMConv2D(filters=filters, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filters, lut_file):
    x_skip = x
    x = AMConv2D(filters=filters, kernel_size=3, strides=2, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = AMConv2D(filters=filters, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Adjust skip connection with stride=2
    x_skip = AMConv2D(filters=filters, kernel_size=1, strides=2, padding='same', mant_mul_lut=lut_file)(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNetCIFAR(shape=(32, 32, 3), classes=10, lut_file="lut/MBM_7.bin"):
    # For CIFAR-10, use a 3x3 conv instead of the 7x7 variant.
    inputs = tf.keras.layers.Input(shape)
    x = AMConv2D(filters=16, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # First group: 2 blocks with 16 filters
    for i in range(2):
        x = identity_block(x, 16, lut_file)
    
    # Second group: 1 convolutional block to downsample then 1 identity block with 32 filters
    x = convolutional_block(x, 32, lut_file)
    x = identity_block(x, 32, lut_file)
    
    # Third group: 1 convolutional block to downsample then 1 identity block with 64 filters
    x = convolutional_block(x, 64, lut_file)
    x = identity_block(x, 64, lut_file)
    
    # Global average pooling and dense layers using custom denseam layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = denseam(128, activation='relu', mant_mul_lut=lut_file)(x)
    outputs = denseam(classes, activation='softmax', mant_mul_lut=lut_file)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="ResNetCIFAR")
    return model

# Native data augmentation using tf.data and tf.image
def augment(image, label):
    # Pad image to 40x40, then randomly crop back to 32x32
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Create tf.data datasets with augmentation for training
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(5000).batch(64).prefetch(tf.data.AUTOTUNE)

# Validation dataset (no augmentation)
val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# Instantiate the model
model = ResNetCIFAR(shape=(32, 32, 3), classes=10)

# Compile the model with SGD (faster convergence on CIFAR-10)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Callback to reduce learning rate when validation accuracy plateaus
lr_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-4, verbose=1)

# Train the model (15 epochs)
model.fit(train_ds,
          epochs=15,
          validation_data=val_ds,
          callbacks=[lr_reduction])

# Evaluate the model
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc}")
