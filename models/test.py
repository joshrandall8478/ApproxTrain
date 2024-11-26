import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
import sys

# Optional: Allow specifying ResNet depth via command-line argument
if len(sys.argv) > 1:
    try:
        resnet_depth = int(sys.argv[1])
        if resnet_depth not in [18, 34, 50]:
            raise ValueError
    except ValueError:
        print("Invalid ResNet depth specified. Choose from 18, 34, 50.")
        sys.exit(1)
else:
    resnet_depth = 18  # Default to ResNet-18

# 1. Initialize MirroredStrategy
print("Initializing MirroredStrategy...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs available: {len(gpus)}")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("No GPUs found. Using default strategy.")
    strategy = tf.distribute.get_strategy()

print(f"Number of replicas: {strategy.num_replicas_in_sync}")

# 2. Define Residual Blocks
def basic_residual_block(inputs, filters, stride=1):
    shortcut = inputs
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def bottleneck_residual_block(inputs, filters, stride=1):
    shortcut = inputs
    # 1x1 Conv
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3x3 Conv
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 1x1 Conv
    x = layers.Conv2D(filters * 4, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters * 4:
        shortcut = layers.Conv2D(filters * 4, kernel_size=1, strides=stride,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# 3. ResNet Builder Function
def build_resnet(input_shape=(32, 32, 3), num_classes=10, depth=18):
    """
    Builds a ResNet model of specified depth.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        depth (int): Depth of the ResNet model (18, 34, 50).

    Returns:
        tf.keras.Model: ResNet model.
    """
    if depth == 18:
        block_fn = basic_residual_block
        layers_per_block = [2, 2, 2, 2]
    elif depth == 34:
        block_fn = basic_residual_block
        layers_per_block = [3, 4, 6, 3]
    elif depth == 50:
        block_fn = bottleneck_residual_block
        layers_per_block = [3, 4, 6, 3]
    else:
        raise ValueError("Unsupported ResNet depth. Choose from 18, 34, 50.")

    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Max Pooling
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Define the number of filters for each stage
    filters = [64, 128, 256, 512]
    
    # Build ResNet stages
    for i in range(4):
        for j in range(layers_per_block[i]):
            stride = 1
            if j == 0 and i != 0:
                stride = 2  # Downsample except for the first stage
            x = block_fn(x, filters[i], stride=stride)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 4. Prepare Data
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Normalizing data...")
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

print("Encoding labels...")
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Shuffle and split the training dataset into training and validation sets
print("Shuffling and splitting data into training and validation sets...")
train_size = int(0.9 * len(x_train))
val_size = len(x_train) - train_size

train_dataset = train_dataset.shuffle(buffer_size=50000, seed=42)
val_dataset = train_dataset.skip(train_size)
train_dataset = train_dataset.take(train_size)

print(f"Training samples: {train_size}, Validation samples: {val_size}")

# Data augmentation function
def augment(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Add padding of 4 pixels on each side and then random crop back to 32x32
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

# Create tf.data.Dataset pipelines
batch_size = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Training dataset with augmentation
train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(AUTOTUNE)

# Validation dataset
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# Test dataset
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# 5. Build and Compile the Model within strategy.scope()
with strategy.scope():
    print(f"Building ResNet-{resnet_depth} model within strategy scope...")
    model = build_resnet(input_shape=(32, 32, 3), num_classes=10, depth=resnet_depth)
    model.summary()
    
    print("Compiling the model...")
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 6. Define Callbacks
print("Setting up callbacks...")
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=10,
                                 factor=0.5,
                                 min_lr=1e-6,
                                 verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=20,
                               verbose=1,
                               restore_best_weights=True)

model_checkpoint = ModelCheckpoint('resnet{}_cifar10.h5'.format(resnet_depth),
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   verbose=1)

# 7. Train the Model
epochs = 100  # Adjust as needed

print("Starting training...")
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[lr_reduction, early_stopping, model_checkpoint],
    verbose=1  # Displays progress bars and metrics in the terminal
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTraining completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds.")

# 8. Evaluate the Model
print("\nLoading the best saved model...")
model.load_weights('resnet{}_cifar10.h5'.format(resnet_depth))

print("Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 9. Generate Predictions and True Labels
print("\nGenerating predictions for detailed evaluation...")
y_pred = []
y_true = []

for batch_images, batch_labels in test_dataset:
    preds = model.predict(batch_images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1))

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# 10. Compute Confusion Matrix
print("\nComputing confusion matrix...")
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
print("Confusion Matrix:")
print(confusion_mtx)

# 11. Calculate Per-Class Metrics
print("\nPer-Class Metrics:")
for i in range(num_classes):
    TP = confusion_mtx[i, i]
    FP = np.sum(confusion_mtx[:, i]) - TP
    FN = np.sum(confusion_mtx[i, :]) - TP
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    print(f"Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
