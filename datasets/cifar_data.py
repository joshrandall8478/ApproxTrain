import tensorflow as tf
import tensorflow_datasets as tfds
# cifar 10 and 100 data loading
def load_cifar_data(dataset_name):
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Split validation set from training data
    val_size = int(0.1 * ds_info.splits['train'].num_examples)
    ds_val = ds_train.take(val_size)
    ds_train = ds_train.skip(val_size)

    # Dataset preparation
    ds_train = ds_train.map(normalize_img).cache().shuffle(45000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    input_shape = (32, 32, 3)
    num_classes = 10 if dataset_name == 'cifar10' else 100

    return ds_train, ds_val, ds_test, input_shape, num_classes
