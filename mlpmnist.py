import tensorflow as tf
import tensorflow_datasets as tfds
from python.keras.layers.amdenselayer import denseam

tf.random.set_seed(0)

# Load and normalize MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

# Path to your LUT for approximate multiplication
lut_file = "lut/MBM_7.bin"

# Define the MLP model using denseam
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    denseam(512, activation='relu', mant_mul_lut=lut_file),
    tf.keras.layers.Dropout(0.3),
    denseam(256, activation='relu', mant_mul_lut=lut_file),
    tf.keras.layers.Dropout(0.3),
    denseam(128, activation='relu', mant_mul_lut=lut_file),
    tf.keras.layers.Dropout(0.3),
    denseam(10, activation='softmax', mant_mul_lut=lut_file)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

