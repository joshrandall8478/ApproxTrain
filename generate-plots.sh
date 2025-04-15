#!/bin/bash

# Array of models
#models=("approx-cifar-10-mlp.py" "approx-cifar-10-resnet-josh.py" "approx-cifar-10-resnet-maksim.py" "approx-cifar-10-vgg.py" "approx-cifar-100-mlp.py" "approx-cifar-100-vgg.py" "approx-mnist-mlp.py")
models=("approx-cifar-10-resnet-josh.py" "approx-cifar-10-resnet-maksim.py" "approx-cifar-10-vgg.py" "approx-cifar-100-mlp.py" "approx-cifar-100-vgg.py" "approx-mnist-mlp.py")

# Directory containing LUT files
lut_dir="lut"

# MBM
# Loop through each model
for model in "${models[@]}"; do
    # Loop through each LUT file in the directory
    for lut_file in "$lut_dir"/MBM_7.bin; do
        python3 "$model" --mul "$lut_file"
    done
done

# MIT
# Loop through each model
for model in "${models[@]}"; do
    # Loop through each LUT file in the directory
    for lut_file in "$lut_dir"/MIT_7.bin; do
        python3 "$model" --mul "$lut_file"
    done
done