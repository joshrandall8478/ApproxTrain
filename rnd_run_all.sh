#!/bin/bash

# -------------------------------------------------------------------
# Script: run_experiments.sh
# Description: Automate running experiments based on a CSV configuration.
# Author: Jing Gong
# -------------------------------------------------------------------

# Note: Removed 'set -e' to handle errors explicitly
# set -e  # Commented out to handle errors explicitly

# ------------------------ Configuration --------------------------

# Path to the CSV file containing experiment configurations
CSV_FILE="experiments.csv"

# Temporary CSV file to store updated results
TEMP_CSV="experiments_temp.csv"

# Path to your Python training script
PYTHON_SCRIPT="train.py"

# ------------------------ Cleanup Function --------------------------

# Function to clean up and move the temp CSV on exit
cleanup() {
    if [[ -f "$TEMP_CSV" ]]; then
        # Append the remaining unprocessed lines to TEMP_CSV
        TOTAL_LINES=$(wc -l < "$CSV_FILE")
        if [[ $CURRENT_LINE -le $TOTAL_LINES ]]; then
            tail -n +"$CURRENT_LINE" "$CSV_FILE" >> "$TEMP_CSV"
        fi
        mv "$TEMP_CSV" "$CSV_FILE"
        echo -e "\nAll experiments processed. Updated CSV saved to '$CSV_FILE'."
    else
        echo "No temporary CSV to save."
    fi
}

# Trap EXIT signal to run the cleanup function
trap cleanup EXIT

# ------------------------ Functions -----------------------------

# Function to extract Test Accuracy and Avg Batch Time from JSON using inline Python
extract_metrics() {
    JSON_FILE=$1
    # Inline Python to parse JSON and extract 'test_accuracy' and 'avg_batch_time'
    METRICS=$(python3 -c "
import json
try:
    with open('$JSON_FILE') as f:
        data = json.load(f)
    test_acc = data.get('test_accuracy', 'N/A')
    avg_batch_time = data.get('avg_batch_time', 'N/A')
    print(f'{test_acc},{avg_batch_time}')
except Exception as e:
    print('N/A,N/A')
")
    echo "$METRICS"
}

# Function to construct the stats JSON file path based on parameters
construct_stats_path() {
    MODEL=$1
    DATASET=$2
    LUT_FILE_NAME=$3
    FPMode=$4
    ROUNDING=$5
    EARLYSTOPPING=$6

    if [[ "$EARLYSTOPPING" == "Yes" || "$EARLYSTOPPING" == "yes" ]]; then
        EARLYSTOPPING_SUFFIX="_earlystopping"
    else
        EARLYSTOPPING_SUFFIX=""
    fi

    STATS_FILE="save/training_stats/${MODEL}_${DATASET}_${LUT_FILE_NAME}_${FPMode}_${ROUNDING}${EARLYSTOPPING_SUFFIX}.json"
    echo "$STATS_FILE"
}

# Function to run the Python training script with given parameters
run_python_script() {
    SEED=$1
    LUT=$2
    EPOCH=$3
    EARLYSTOPPING=$4
    ROUNDING=$5
    FPMode=$6
    MODEL=$7
    DATASET=$8

    # Handle LUT parameter
    if [[ "$LUT" == "No" || "$LUT" == "no" ]]; then
        LUT_ARG=""
        LUT_FILE_NAME="default"
    else
        LUT_ARG="--LUT $LUT"
        # Extract the base name without extension
        LUT_FILE_NAME=$(basename "$LUT" .bin)
    fi

    # Handle EARLYSTOPPING parameter
    if [[ "$EARLYSTOPPING" == "Yes" || "$EARLYSTOPPING" == "yes" ]]; then
        EARLYSTOPPING_ARG="--EARLYSTOPPING"
    else
        EARLYSTOPPING_ARG=""
    fi

    # Construct the command
    CMD="python3 $PYTHON_SCRIPT --SEED $SEED"

    # Conditionally add --LUT parameter
    if [[ -n "$LUT_ARG" ]]; then
        CMD="$CMD $LUT_ARG"
    fi

    CMD="$CMD --EPOCH $EPOCH $EARLYSTOPPING_ARG --ROUNDING $ROUNDING --FPMode $FPMode --MODEL $MODEL --DATASET $DATASET"

    # Remove any double spaces caused by empty arguments
    CMD=$(echo "$CMD" | tr -s ' ')

    echo "Executing: $CMD"

    # Execute the Python script and capture the exit status
    eval "$CMD"
    STATUS=$?

    if [[ $STATUS -ne 0 ]]; then
        echo "Error: Python script failed with status $STATUS."
        # Return non-zero status to the caller
        return $STATUS
    fi
}

# ------------------------ Main Script ----------------------------

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

# Ensure the CSV file ends with a newline
if [ -n "$(tail -c1 "$CSV_FILE")" ] && [ "$(tail -c1 "$CSV_FILE")" != $'\n' ]; then
    echo "" >> "$CSV_FILE"
fi

# Remove any trailing empty lines
sed -i '/^$/d' "$CSV_FILE"

# Read the header line
HEADER=$(head -n 1 "$CSV_FILE")
echo "$HEADER" > "$TEMP_CSV"

# Initialize line counter
CURRENT_LINE=2  # Start from line 2 because header is line 1

# Read the CSV excluding the header using process substitution
while IFS=, read -r SEED LUT EPOCH EARLYSTOPPING ROUNDING FPMode MODEL DATASET TRAIN_TIME TEST_ACC
do
    # Trim whitespace from variables
    SEED=$(echo "$SEED" | xargs)
    LUT=$(echo "$LUT" | xargs)
    EPOCH=$(echo "$EPOCH" | xargs)
    EARLYSTOPPING=$(echo "$EARLYSTOPPING" | xargs)
    ROUNDING=$(echo "$ROUNDING" | xargs)
    FPMode=$(echo "$FPMode" | xargs)
    MODEL=$(echo "$MODEL" | xargs)
    DATASET=$(echo "$DATASET" | xargs)
    TRAIN_TIME=$(echo "$TRAIN_TIME" | xargs)
    TEST_ACC=$(echo "$TEST_ACC" | xargs)

    # Check if 'Train time per batch' and 'Test Acc' are empty
    if [[ -z "$TRAIN_TIME" && -z "$TEST_ACC" ]]; then
        echo -e "\n=== Running Experiment ==="
        echo "Model: $MODEL"
        echo "Dataset: $DATASET"
        echo "FPMode: $FPMode"
        echo "Seed: $SEED"
        echo "LUT: $LUT"
        echo "Epoch: $EPOCH"
        echo "EarlyStopping: $EARLYSTOPPING"
        echo "Rounding: $ROUNDING"

        # Run the Python script
        run_python_script "$SEED" "$LUT" "$EPOCH" "$EARLYSTOPPING" "$ROUNDING" "$FPMode" "$MODEL" "$DATASET"
        RUN_STATUS=$?

        if [[ $RUN_STATUS -ne 0 ]]; then
            echo "Error: Python script failed with status $RUN_STATUS."
            echo "Skipping metrics extraction and recording."
            # Append the row without updating metrics
            echo "$SEED,$LUT,$EPOCH,$EARLYSTOPPING,$ROUNDING,$FPMode,$MODEL,$DATASET,," >> "$TEMP_CSV"
            # Optionally, you can choose to exit or continue with the next experiment
            # For this script, we'll continue with the next experiment
            # Increment the line counter
            ((CURRENT_LINE++))
            continue
        else
            # Determine LUT file name
            if [[ "$LUT" == "No" || "$LUT" == "no" ]]; then
                LUT_FILE_NAME="default"
            else
                LUT_FILE_NAME=$(basename "$LUT" .bin)
            fi

            # Construct the path to the JSON stats file
            STATS_FILE=$(construct_stats_path "$MODEL" "$DATASET" "$LUT_FILE_NAME" "$FPMode" "$ROUNDING" "$EARLYSTOPPING")

            # Check if the JSON stats file exists
            if [[ -f "$STATS_FILE" ]]; then
                echo "Found stats file: $STATS_FILE"

                # Extract Test Accuracy and Avg Batch Time using inline Python
                METRICS=$(extract_metrics "$STATS_FILE")
                TEST_ACC_VALUE=$(echo "$METRICS" | cut -d',' -f1)
                TRAIN_TIME_PER_BATCH=$(echo "$METRICS" | cut -d',' -f2)

                echo "Train Time per Batch: $TRAIN_TIME_PER_BATCH seconds"
                echo "Test Accuracy: $TEST_ACC_VALUE"

                # Append the updated row to the temporary CSV
                echo "$SEED,$LUT,$EPOCH,$EARLYSTOPPING,$ROUNDING,$FPMode,$MODEL,$DATASET,$TRAIN_TIME_PER_BATCH,$TEST_ACC_VALUE" >> "$TEMP_CSV"
            else
                echo "Error: Stats file '$STATS_FILE' not found. Skipping metrics recording."
                # Append the row without updating metrics
                echo "$SEED,$LUT,$EPOCH,$EARLYSTOPPING,$ROUNDING,$FPMode,$MODEL,$DATASET,," >> "$TEMP_CSV"
            fi
        fi
    else
        echo -e "\n=== Skipping Experiment ==="
        echo "Model: $MODEL"
        echo "Dataset: $DATASET"
        echo "FPMode: $FPMode"
        echo "Seed: $SEED"
        echo "LUT: $LUT"
        echo "Epoch: $EPOCH"
        echo "EarlyStopping: $EARLYSTOPPING"
        echo "Rounding: $ROUNDING"
        echo "Results already present. Skipping."

        # Append the existing row to the temporary CSV
        echo "$SEED,$LUT,$EPOCH,$EARLYSTOPPING,$ROUNDING,$FPMode,$MODEL,$DATASET,$TRAIN_TIME,$TEST_ACC" >> "$TEMP_CSV"
    fi

    # Increment the line counter
    ((CURRENT_LINE++))

done < <(tail -n +2 "$CSV_FILE")

# Note: The cleanup function will be called automatically due to the 'trap' command
