import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam


#VGG 16   uses cfar10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

import argparse
parser = argparse.ArgumentParser(description='Path to the LUT file')
parser.add_argument('--mul', type=str, required=True, help='Path to the LUT file')
args = parser.parse_args()
lut_file = args.mul

print("Lut file: " + lut_file)


def vgg(input_shape=(32,32,3),num_classes=100):
    model = tf.keras.Sequential()

# block 1

    model.add(AMConv2D(filters=64, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(AMConv2D(filters=64, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Block 2
    model.add(AMConv2D(filters=128, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(AMConv2D(filters=128, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Block 3
    model.add(AMConv2D(filters=256, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(AMConv2D(filters=256, kernel_size=3, padding='same', activation='relu', mant_mul_lut=lut_file))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(denseam(1024, activation='relu', mant_mul_lut=lut_file))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(denseam(100, activation='softmax', mant_mul_lut=lut_file))

    return model




#compile model

model = vgg()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=12, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

import os
import matplotlib.pyplot as plt

# Plot training and validation accuracy
history = model.history.history
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('VGG Cifar 10 - ' + lut_file)
plt.legend()
# plt.show()
# Save the plot as a PNG file to the "plots" directory

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

# Save the plot
plot_filename = "plots/vgg_cifar_10_" + os.path.basename(lut_file) + ".png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")