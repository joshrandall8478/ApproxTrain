import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Adjust y_train and y_test to ensure they are properly formatted for training
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10).astype('float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10).astype('float32')


lut_file = "lut/MBM_7.bin"

from tensorflow.keras.initializers import HeNormal
# Residual Block with Approximate Conv2D

def residual_block(inputs, filters, stride=1, use_downsample=False):
    x = AMConv2D(filters, kernel_size=3, strides=stride, padding='same', activation='swish', 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)

    x = AMConv2D(filters, kernel_size=3, padding='same', activation=None, 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = inputs
    if use_downsample:
        shortcut = AMConv2D(filters, kernel_size=1, strides=stride, padding='same', activation=None, 
                            kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])

    x = tf.keras.layers.Activation('swish')(x)
    return x


# Build ResNet-like Model
def resnet(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial Convolutional Layer
    x = AMConv2D(64, kernel_size=3, strides=1, padding='same', activation='swish', 
                 kernel_initializer=HeNormal(), mant_mul_lut=lut_file)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual Blocks
    x = residual_block(x, 64)
    # x = residual_block(x, 64)

    # # Add more residual blocks with increasing filters
    # x = residual_block(x, 128, stride=2, use_downsample=True)
    # x = residual_block(x, 128)
    # x = residual_block(x, 256, stride=2, use_downsample=True)
    # x = residual_block(x, 256)
    # x = residual_block(x, 512, stride=2, use_downsample=True)
    # x = residual_block(x, 512)

    # Flatten and add Dense layers
    x = tf.keras.layers.Flatten()(x)
    # x = denseam(256, activation='relu', mant_mul_lut=lut_file)(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularization
    x = tf.keras.layers.BatchNormalization()(x)

    # Dense output layer
    outputs = denseam(num_classes, activation='softmax', mant_mul_lut=lut_file)(x)

    return tf.keras.Model(inputs, outputs)


# Define learning rate schedule
# lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#     initial_learning_rate=0.01,  # Increased initial learning rate
#     first_decay_steps=10000,    # Increased decay steps for smoother decay
#     t_mul=2.0,
#     m_mul=0.8,                  # Adjusted multiplier for more gradual decay
#     alpha=0.00001               # Lowered alpha for a smaller final learning rate
# )

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,  # Start with a smaller learning rate
    first_decay_steps=1000,      # Adjust decay steps for smoother learning
    t_mul=2.0,
    m_mul=0.8,                   # Gradual decay multiplier
    alpha=0.0001                 # Final learning rate
)

# Create Adam optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0 )

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile and Train
model = resnet()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)