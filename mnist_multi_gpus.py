import argparse
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a model with customizable SEED, LUT file, EPOCH, EARLYSTOPPING, and FP16 support.")
parser.add_argument("--SEED", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--LUT", type=str, help="Path to the LUT file (ignored if FP16 is enabled)")
parser.add_argument("--EPOCH", type=int, default=20, help="Number of training epochs")
parser.add_argument("--EARLYSTOPPING", action='store_true', help="Enable early stopping")
parser.add_argument("--ROUNDING", type=str, required=True, help="Rounding mode for data collection")
parser.add_argument("--FP16", action='store_true', help="Use FP16 mode (ignores LUT files)")
args = parser.parse_args()

# Check that LUT is provided unless FP16 is enabled
if not args.FP16 and not args.LUT:
    parser.error("Argument --LUT is required unless --FP16 is specified")

# Determine LUT filename or set to "FP16" if FP16 mode is enabled
if args.FP16:
    lut_file_name = "FP16"
    lut_file = "lut/ZEROS_7.bin"
else:
    lut_file_name = re.match(r"(.+)\.bin$", os.path.basename(args.LUT)).group(1) if args.LUT else "default"
    lut_file = args.LUT

rnd = args.ROUNDING

# Set random seed
tf.random.set_seed(args.SEED)

# Set up data loading with train, validation, and test splits
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train[:80%]', 'train[80%:]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label



# Prepare the training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(int(ds_info.splits['train'].num_examples * 0.8))
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Prepare the validation dataset
ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.batch(128)
ds_val = ds_val.cache()
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

# Prepare the test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Set up multi-GPU training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut=lut_file),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        denseam(1024, activation='relu', mant_mul_lut=lut_file),
        tf.keras.layers.Dropout(0.4),
        denseam(10, activation='softmax', mant_mul_lut=lut_file)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

# Ensure directories exist for saving checkpoints and statistics
os.makedirs("save/checkpoints", exist_ok=True)
os.makedirs("save/training_stats", exist_ok=True)

# Define checkpoint file path based on LUT filename or FP16 mode
checkpoint_callback_file_path = f"save/checkpoints/lenet5_mnist_{lut_file_name}_{rnd}.h5"
if args.EARLYSTOPPING:
    checkpoint_callback_file_path = f"save/checkpoints/lenet5_mnist_{lut_file_name}_earlystopping_{rnd}.h5"

# Define a callback for saving the best model based on validation accuracy
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_callback_file_path,
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Define early stopping callback if EARLYSTOPPING is enabled
callbacks = [checkpoint_callback]
if args.EARLYSTOPPING:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)

# Train the model with distributed strategy and callbacks
history = model.fit(
    ds_train,
    epochs=args.EPOCH,
    validation_data=ds_val,
    callbacks=callbacks,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save training statistics to a JSON file
training_stats = {
    "train_loss": history.history['loss'],
    "train_accuracy": history.history['sparse_categorical_accuracy'],
    "val_loss": history.history['val_loss'],
    "val_accuracy": history.history['val_sparse_categorical_accuracy'],
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
}

# Define training statistics file path
training_stats_file_path = f"save/training_stats/lenet5_mnist_{lut_file_name}_{rnd}.json"
if args.EARLYSTOPPING:
    training_stats_file_path = f"save/training_stats/lenet5_mnist_{lut_file_name}_earlystopping_{rnd}.json"

with open(training_stats_file_path, "w") as f:
    json.dump(training_stats, f)

print(f"Training statistics saved to {training_stats_file_path}")
