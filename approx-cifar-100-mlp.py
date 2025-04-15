import tensorflow as tf
import tensorflow_datasets as tfds
from python.keras.layers.amdenselayer import denseam

tf.random.set_seed(0)

# Load CIFAR-100 dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cifar100',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalize the image to [0, 1]."""
    return tf.cast(image, tf.float32) / 255., label

# Preprocess
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

import argparse
parser = argparse.ArgumentParser(description='Path to the LUT file')
parser.add_argument('--mul', type=str, required=True, help='Path to the LUT file')
args = parser.parse_args()
lut_file = args.mul

print("Lut file: " + lut_file)

# MLP model for CIFAR-100
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),

    denseam(2048, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),
    tf.keras.layers.Dropout(0.2),

    denseam(1024, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),
    tf.keras.layers.Dropout(0.2),

    denseam(512, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),
    tf.keras.layers.Dropout(0.2),

    denseam(100, activation='softmax', mant_mul_lut=lut_file)  # 100 classes for CIFAR-100
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train
model.fit(
    ds_train,
    epochs=12,
    validation_data=ds_test,
)


import os
import matplotlib.pyplot as plt

# Plot training and validation accuracy
history = model.history.history
plt.plot(history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('MLP Cifar 100 - ' + lut_file)
plt.legend()
# plt.show()
# Save the plot as a PNG file to the "plots" directory

# Ensure the "plots" directory exists
os.makedirs("plots", exist_ok=True)

# Save the plot
plot_filename = "plots/mlp_cifar_100_" + os.path.basename(lut_file) + ".png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")