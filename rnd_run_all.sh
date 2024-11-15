#!/bin/bash

# Default values
SEED=0
EPOCH=20
EARLYSTOPPING="False"
RND="RTZ"
FP16="False"
lut_files=("lut/MBM_7.bin" "lut/MIT_7.bin" "lut/ZEROS_7.bin")

# Function to print usage
usage() {
    echo "Usage: $0 [-s SEED] [-e EPOCH] [-t] [-r ROUNDING_MODE] [-f]"
    echo "  -s SEED               Integer seed value (default: 0)"
    echo "  -e EPOCH              Integer epoch count (default: 20)"
    echo "  -t                    Enable early stopping"
    echo "  -r ROUNDING_MODE      Rounding mode: RTZ or RNE (default: RTZ)"
    echo "  -f                    Use FP16 mode (ignores LUT files)"
    exit 1
}

# Parse arguments
while getopts "s:e:tr:f" opt; do
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
        t ) EARLYSTOPPING="True"
            ;;
        r ) RND=$OPTARG
            if [[ "$RND" != "RTZ" && "$RND" != "RNE" ]]; then
                echo "Error: ROUNDING_MODE must be 'RTZ' or 'RNE'."
                usage
            fi
            ;;
        f ) FP16="True"
            ;;
        * ) usage
            ;;
    esac
done

# Run experiment based on FP16 mode
if [[ "$FP16" == "True" ]]; then
    echo "Running single experiment with FP16 mode, SEED: $SEED, EPOCH: $EPOCH, EARLYSTOPPING: $EARLYSTOPPING, ROUNDING_MODE: $RND"
    cmd="python mnist_multi_gpus.py --SEED \"$SEED\" --EPOCH \"$EPOCH\" --ROUNDING \"$RND\" --FP16"
    if [[ "$EARLYSTOPPING" == "True" ]]; then
        cmd+=" --EARLYSTOPPING"
    fi
    echo "Executing command: $cmd"
    eval $cmd
else
    # Loop over each LUT file and execute the Python script if FP16 is not specified
    for lut_file in "${lut_files[@]}"; do
        echo "Running experiment with LUT file: $lut_file, SEED: $SEED, EPOCH: $EPOCH, EARLYSTOPPING: $EARLYSTOPPING, ROUNDING_MODE: $RND"
        cmd="python mnist_multi_gpus.py --SEED \"$SEED\" --LUT \"$lut_file\" --EPOCH \"$EPOCH\" --ROUNDING \"$RND\""
        if [[ "$EARLYSTOPPING" == "True" ]]; then
            cmd+=" --EARLYSTOPPING"
        fi
        echo "Executing command: $cmd"
        eval $cmd
    done
fi

echo "All experiments completed."
