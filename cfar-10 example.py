import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam


#VGG 16   uses cfar10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

lut_file = "lut/MBM_7.bin"


def vgg(input_shape=(32,32,3),num_classes=10):
    model = tf.keras.Sequential()

# block 1

    model.add(AMConv2D(filters=64, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(AMConv2D(filters=64, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Block 2
    model.add(AMConv2D(filters=128, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(AMConv2D(filters=128, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Block 3
    model.add(AMConv2D(filters=256, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(AMConv2D(filters=256, kernel_size=3, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(denseam(10, activation='softmax', mant_mul_lut=lut_file))

    return model




#compile model

model = vgg()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)