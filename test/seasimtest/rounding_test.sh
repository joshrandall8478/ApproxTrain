#!/bin/bash

# Define the source file and the output executable
SOURCE_FILE="rounding.cu"
EXECUTABLE="rounding_test"

# Step 1: Compile the CUDA program
echo "Compiling $SOURCE_FILE..."
nvcc $SOURCE_FILE -o $EXECUTABLE

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation successful."

# Step 2: Run the compiled program
echo "Running $EXECUTABLE..."
./$EXECUTABLE

# Step 3: Clean up by removing the executable (optional)
# Uncomment the following line if you want to delete the executable after running
# rm $EXECUTABLE
