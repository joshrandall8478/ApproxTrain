import argparse
import os
import re
import tensorflow as tf
import json

from models.lenet import build_lenet
from models.resnet import build_resnet
from datasets.mnist_data import load_mnist_data
from datasets.cifar_data import load_cifar_data

from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a model with customizable options.")
    parser.add_argument("--SEED", type=int, default=0, help="Random seed")
    parser.add_argument("--LUT", type=str, help="Path to the LUT file")
    parser.add_argument("--EPOCH", type=int, default=20, help="Number of epochs")
    parser.add_argument("--EARLYSTOPPING", action='store_true', help="Enable early stopping")
    parser.add_argument("--ROUNDING", type=str, required=True, help="Rounding mode")
    parser.add_argument("--FP16", action='store_true', help="Use FP16 mode")
    parser.add_argument("--FP8", action='store_true', help="Use FP8 mode")
    parser.add_argument("--MODEL", type=str, required=True, choices=['lenet300100', 'lenet5', 'resnet18', 'resnet34', 'resnet50'], help="Model architecture to train")
    parser.add_argument("--DATASET", type=str, required=True, choices=['mnist', 'cifar10', 'cifar100'], help="Dataset to use for training")
    args = parser.parse_args()

    # Check LUT requirement
    if not (args.FP16 or args.FP8) and not args.LUT:
        parser.error("Argument --LUT is required unless --FP16 or --FP8 is specified")
    if args.FP16 and args.FP8:
        parser.error("Cannot specify both --FP16 and --FP8")

    # Determine LUT file
    if args.FP16:
        lut_file_name = "FP16"
        lut_file = "lut/ZEROS_7.bin"
    elif args.FP8:
        lut_file_name = "FP8"
        lut_file = "lut/FP8/combined_fp8_mul_lut.bin"
    else:
        lut_file_name = re.match(r"(.+)\.bin$", os.path.basename(args.LUT)).group(1) if args.LUT else "default"
        lut_file = args.LUT

    rnd = args.ROUNDING
    tf.random.set_seed(args.SEED)

    # Data loading
    if args.DATASET == 'mnist':
        ds_train, ds_val, ds_test, input_shape, num_classes = load_mnist_data()
    elif args.DATASET in ['cifar10', 'cifar100']:
        ds_train, ds_val, ds_test, input_shape, num_classes = load_cifar_data(args.DATASET)
    else:
        raise ValueError(f"Unsupported dataset: {args.DATASET}")

    # Model setup
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.MODEL.startswith('lenet'):
            model = build_lenet(args.MODEL, input_shape, num_classes, lut_file, args.FP8)
        elif args.MODEL.startswith('resnet'):
            model = build_resnet(args.MODEL, input_shape, num_classes, lut_file, args.FP8)
        else:
            raise ValueError(f"Unsupported model: {args.MODEL}")

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Directories
    os.makedirs("save/checkpoints", exist_ok=True)
    os.makedirs("save/training_stats", exist_ok=True)

    # Callbacks
    checkpoint_path = f"save/checkpoints/{args.MODEL}_{args.DATASET}_{lut_file_name}_{rnd}.h5"
    if args.EARLYSTOPPING:
        checkpoint_path = f"save/checkpoints/{args.MODEL}_{args.DATASET}_{lut_file_name}_earlystopping_{rnd}.h5"
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
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)

    # Training
    history = model.fit(
        ds_train,
        epochs=args.EPOCH,
        validation_data=ds_val,
        callbacks=callbacks,
    )

    # Evaluation
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Save stats
    training_stats = {
        "train_loss": history.history['loss'],
        "train_accuracy": history.history['accuracy'],
        "val_loss": history.history['val_loss'],
        "val_accuracy": history.history['val_accuracy'],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    stats_path = f"save/training_stats/{args.MODEL}_{args.DATASET}_{lut_file_name}_{rnd}.json"
    if args.EARLYSTOPPING:
        stats_path = f"save/training_stats/{args.MODEL}_{args.DATASET}_{lut_file_name}_earlystopping_{rnd}.json"
    with open(stats_path, "w") as f:
        json.dump(training_stats, f)
    print(f"Training statistics saved to {stats_path}")

if __name__ == "__main__":
    main()
