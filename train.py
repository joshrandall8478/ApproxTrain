# train.py

import argparse
import os
import re
import tensorflow as tf
import json
import time

from models.lenet import build_lenet  
from models.resnet import build_resnet_cifar
from datasets.mnist_data import load_mnist_data  # Keep MNIST unchanged
from datasets.cifar_data import load_cifar_data

from tensorflow.keras.callbacks import ReduceLROnPlateau

class BatchTimeLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_times = []
        self.batch_start_time = None

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)

    def on_train_end(self, logs=None):
        if self.batch_times:
            self.avg_batch_time = sum(self.batch_times) / len(self.batch_times)
        else:
            self.avg_batch_time = 0.0

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a model with customizable options.")
    parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    parser.add_argument("--LUT", type=str, help="Path to the LUT file")
    parser.add_argument("--EPOCH", type=int, default=200, help="Number of epochs")
    parser.add_argument("--EARLYSTOPPING", action='store_true', help="Enable early stopping")
    parser.add_argument("--ROUNDING", type=str, required=True, choices=['RNE','RTZ'], help="Rounding mode (you have to recompile ApproxTrain if you change the rounding mode)")
    # Add a new argument to specify the FPMode
    parser.add_argument("--FPMode", type=str, required=True, choices=['FP32', 'FP16', 'BF16', 'FP8E5M2', 'FP8HYB'], help="FPMode to use for training")
    parser.add_argument("--MODEL", type=str, required=True, choices=['lenet300100', 'lenet5', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202'], help="Model architecture to train")
    parser.add_argument("--DATASET", type=str, required=True, choices=['mnist', 'cifar10', 'cifar100'], help="Dataset to use for training")
    args = parser.parse_args()
    
    lut_file_name = re.match(r"(.+)\.bin$", os.path.basename(args.LUT)).group(1) if args.LUT else "default"
    lut_file = args.LUT

    rnd = args.ROUNDING
    tf.random.set_seed(args.SEED)

    # Data loading
    if args.DATASET == 'mnist':
        ds_train, ds_val, ds_test, input_shape, num_classes = load_mnist_data()
    elif args.DATASET in ['cifar10', 'cifar100']:
        ds_train, ds_val, input_shape, num_classes = load_cifar_data(args.DATASET)
    else:
        raise ValueError(f"Unsupported dataset: {args.DATASET}")

    # Model setup
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.MODEL.startswith('lenet'):
            if args.MODEL == 'lenet5':
                model = build_lenet5(input_shape, num_classes, lut_file, args.FPMode)
            else:
                raise ValueError(f"Unsupported LeNet model: {args.MODEL}")
            optimizer = tf.keras.optimizers.Adam()
        elif args.MODEL.startswith('resnet'):
            depth = int(args.MODEL.replace('resnet', ''))
            model = build_resnet_cifar(input_shape=input_shape, num_classes=num_classes, depth=depth)
            # Use SGD with momentum
            initial_lr = 0.1
            optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unsupported model: {args.MODEL}")
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        model.summary()
    
    test=False
    if test:
        # load the model specified by the user
        model.load_weights(f"save/checkpoints/{args.MODEL}_{args.DATASET}_{lut_file_name}_{args.FPMode}_{rnd}.h5")
        model.evaluate(ds_test)
        return

    # Directories
    os.makedirs("save/checkpoints", exist_ok=True)
    os.makedirs("save/training_stats", exist_ok=True)

    # Callbacks
    checkpoint_path = f"save/checkpoints/{args.MODEL}_{args.DATASET}_{lut_file_name}_{args.FPMode}_{rnd}.h5"
    if args.EARLYSTOPPING:
        checkpoint_path = f"save/checkpoints/{args.MODEL}_{args.DATASET}_{lut_file_name}_{args.FPMode}_{rnd}_earlystopping.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    callbacks = [checkpoint_callback]
    if args.EARLYSTOPPING:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
    batch_time_logger = BatchTimeLogger()
    callbacks.append(batch_time_logger)
    print("Setting up callbacks...")

    # Define learning rate schedule
    def lr_schedule(epoch):
        lr = 0.1
        if epoch >= 150:
            lr *= 0.01
        elif epoch >= 100:
            lr *= 0.1
        print('Learning rate: ', lr)
        return lr
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks.append(lr_scheduler)

    # Training
    history = model.fit(
        ds_train,
        epochs=args.EPOCH,
        validation_data=ds_val,
        callbacks=callbacks,
        steps_per_epoch=ds_train.n // ds_train.batch_size,
        validation_steps=ds_val.n // ds_val.batch_size
    )

    # Evaluation
    # For CIFAR datasets, reuse ds_val as ds_test since they are the same
    test_loss, test_accuracy = model.evaluate(ds_val, steps=ds_val.n // ds_val.batch_size)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Save stats
    training_stats = {
        "train_loss": history.history['loss'],
        "train_accuracy": history.history['accuracy'],
        "val_loss": history.history['val_loss'],
        "val_accuracy": history.history['val_accuracy'],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "avg_batch_time": batch_time_logger.avg_batch_time
    }
    stats_path = f"save/training_stats/{args.MODEL}_{args.DATASET}_{lut_file_name}_{args.FPMode}_{rnd}.json"
    if args.EARLYSTOPPING:
        stats_path = f"save/training_stats/{args.MODEL}_{args.DATASET}_{lut_file_name}_{args.FPMode}_{rnd}_earlystopping.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f)
    print(f"Training statistics saved to {stats_path}")

if __name__ == "__main__":
    main()
