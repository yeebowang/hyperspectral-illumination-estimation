#!/bin/bash

# Check if all the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 initial_epoch max_epoch train_py_file"
    exit 1
fi

# Get the command-line arguments
initial_epoch=$1
MAX_EPOCH=$2
train_py_file=$3

# Function to run the train.py script
run_train() {
    epoch=$1
    python $train_py_file --epoch $epoch
}

# Loop to run the training until the maximum epoch is reached or an error occurs
while [ $initial_epoch -lt $MAX_EPOCH ]; do
    echo "Running $train_py_file with epoch: $initial_epoch"
    run_train $initial_epoch

    # Check the exit status of the train.py script
    if [ $? -eq 0 ]; then
        # If the return value is 0, training was successful, increment the epoch
        initial_epoch=$((initial_epoch + 1))
    else
        # If the return value is not 0, training failed, continue training for the current epoch
        echo "Training failed for epoch: $initial_epoch, retrying..."
    fi
done

echo "Training completed successfully."

