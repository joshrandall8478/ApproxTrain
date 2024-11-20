import tensorflow as tf
import tensorflow_datasets as tfds

def load_mnist_data():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[:80%]', 'train[80%:]', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Dataset preparation
    ds_train = ds_train.map(normalize_img).cache().shuffle(48000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    input_shape = (28, 28, 1)
    num_classes = 10

    return ds_train, ds_val, ds_test, input_shape, num_classes
