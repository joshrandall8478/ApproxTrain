#!/bin/bash

# Default values
SEED=0
EPOCH=20
EARLYSTOPPING="True"
RND="RTZ"
lut_files=("lut/MBM_7.bin" "lut/MIT_7.bin" "lut/ZEROS_7.bin")

# Function to print usage
usage() {
    echo "Usage: $0 [-s SEED] [-e EPOCH] [-t EARLYSTOPPING] [-r ROUNDING_MODE]"
    echo "  -s SEED               Integer seed value (default: 0)"
    echo "  -e EPOCH              Integer epoch count (default: 20)"
    echo "  -t EARLYSTOPPING      Enable early stopping: True or False (default: True)"
    echo "  -r ROUNDING_MODE      Rounding mode: RTZ or RNE (default: RTZ)"
    exit 1
}

# Parse arguments
while getopts "s:e:t:r:" opt; do
    case ${opt} in
        s ) SEED=$OPTARG
            if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
                echo "Error: SEED must be an integer."
                usage
            fi
            ;;
        e ) EPOCH=$OPTARG
            if ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
                echo "Error: EPOCH must be an integer."
                usage
            fi
            ;;
        t ) EARLYSTOPPING=$OPTARG
            if [[ "$EARLYSTOPPING" != "True" && "$EARLYSTOPPING" != "False" ]]; then
                echo "Error: EARLYSTOPPING must be 'True' or 'False'."
                usage
            fi
            ;;
        r ) RND=$OPTARG
            if [[ "$RND" != "RTZ" && "$RND" != "RNE" ]]; then
                echo "Error: ROUNDING_MODE must be 'RTZ' or 'RNE'."
                usage
            fi
            ;;
        * ) usage
            ;;
    esac
done

# Loop over each LUT file and execute the Python script
for lut_file in "${lut_files[@]}"; do
    echo "Running experiment with LUT file: $lut_file, SEED: $SEED, EPOCH: $EPOCH, EARLYSTOPPING: $EARLYSTOPPING, ROUNDING_MODE: $RND"
    python mnist_multi_gpus.py --SEED "$SEED" --LUT "$lut_file" --EPOCH "$EPOCH" --EARLYSTOPPING "$EARLYSTOPPING" --ROUNDING "$RND"
done

echo "All experiments completed."


