import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
tf.random.set_seed(0)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

lut_file = "lut/MBM_7.bin"

from tensorflow.keras.initializers import HeNormal

# Residual Block with Approximate Conv2D
def residual_block(inputs, filters, stride=1, use_downsample=False):
    x = AMConv2D(filters, kernel_size=3, strides=stride, padding='same', activation='relu', 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)
    
    x = AMConv2D(filters, kernel_size=3, padding='same', activation=None, 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(x)
    
    shortcut = inputs
    if use_downsample:
        shortcut = AMConv2D(filters, kernel_size=1, strides=stride, padding='same', activation=None, 
                            kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# Build ResNet-like Model
def resnet(input_shape=(28, 28, 1), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = AMConv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', mant_mul_lut=lut_file)(inputs)

    # Residual Blocks
    x = residual_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = denseam(num_classes, activation='softmax', mant_mul_lut=lut_file)(x)

    return tf.keras.Model(inputs, outputs)


# Normalize and prepare the dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(64)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(64)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=5000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0001
)

# Create Adam optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Compile and Train
model = resnet()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

model.fit(ds_train, epochs=1, validation_data=ds_test)

# Evaluate the model
test_loss, test_acc = model.evaluate(ds_test)
print("Test accuracy:", test_acc)
