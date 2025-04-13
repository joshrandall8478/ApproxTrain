import tensorflow as tf
import tensorflow_datasets as tfds
from python.keras.layers.amdenselayer import denseam

# Set the seed for reproducibility
tf.random.set_seed(0)

# Load CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalize the image to [0, 1]."""
    return tf.cast(image, tf.float32) / 255., label

# Preprocessing and batching
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

# LUT file for approximate multiplier (assuming it's the same as before)
lut_file = "lut/MBM_7.bin"

# Define the MLP model using denseam (for CIFAR-10)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),

    denseam(2048, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),  # ← Swish here
    tf.keras.layers.Dropout(0.2),

    denseam(1024, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),
    tf.keras.layers.Dropout(0.2),

    denseam(512, activation=None, mant_mul_lut=lut_file),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(tf.nn.swish),
    tf.keras.layers.Dropout(0.2),

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
    epochs=20,
    validation_data=ds_test,
)

