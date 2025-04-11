# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
#from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
def identity_block(x, filter, lut_file):
    # Copy tensor to variable called x_skip
    x_skip = x

    # Layer 1: Custom AMConv2D
    x = AMConv2D(filters=filter, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    
    # Layer 2: Custom AMConv2D
    x = AMConv2D(filters=filter, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)

    return x

def convolutional_block(x, filter, lut_file):
    # Save the input tensor to x_skip for the residual connection
    x_skip = x

    # Layer 1: Custom AMConv2D
    x = AMConv2D(filters=filter, kernel_size=3, strides=2, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Layer 2: Custom AMConv2D
    x = AMConv2D(filters=filter, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Match dimensions for the skip connection (use a Conv2D if necessary)
    x_skip = AMConv2D(filters=filter, kernel_size=1, strides=2, padding='same', mant_mul_lut=lut_file)(x_skip)

    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)

    return x

def ResNet34(shape=(32, 32, 3), classes=10, lut_file="lut/MBM_7.bin"):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    
    # Step 2 (Initial Conv layer along with maxPool) using custom AMConv2D
    x = AMConv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    
    # Step 3 Add the ResNet Blocks using custom layers
    for i in range(4):
        if i == 0:
            # For sub-block 1, use identity blocks (no convolutional block)
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size, lut_file)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size, lut_file=lut_file)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size, lut_file)

    # Step 4 End Dense Network using custom denseam layers
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    #x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = denseam(512, activation='relu', mant_mul_lut=lut_file)(x)  # Custom denseam layer
    x = denseam(classes, activation='softmax', mant_mul_lut=lut_file)(x)  # Custom denseam layer
    
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model

#############################################################################
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate the model
model = ResNet34(shape=(32, 32, 3), classes=10)

# Print model summary
#model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

