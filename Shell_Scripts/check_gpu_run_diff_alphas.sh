#!/bin/bash

alpha_ds=(1)
alpha_gs=(32 64 128 256)

# Required free memory in MiB
REQUIRED_FREE_MEMORY=20000  # Adjust this to your needs

# Path to your training script
TRAINING_SCRIPT_PATH="MRIALPHA128DataLoader.py"

# GPU index to check (0 for the first GPU)
GPU_INDEX=0

# Function to check available GPU memory on a specific GPU
check_gpu_memory() {
    AVAILABLE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_INDEX | awk '{print $1}')
    return $AVAILABLE_MEMORY
}

while true; do
    check_gpu_memory
    AVAILABLE_MEMORY=$?

    echo "Available GPU memory on GPU $GPU_INDEX: $AVAILABLE_MEMORY MiB"

    if [ "$AVAILABLE_MEMORY" -ge "$REQUIRED_FREE_MEMORY" ]; then
        echo "Enough GPU memory available on GPU $GPU_INDEX. Starting training."
        for alpha_d in "${alpha_ds[@]}"
        do
                for alpha_g in "${alpha_gs[@]}"
                do
                        echo "Running experiment with alpha_d=${alpha_d} and alpha_g=${alpha_g}"
                        python MRIALPHA128DataLoader.py --alpha_d ${alpha_d} --alpha_g ${alpha_g}
                done
        done
        break
    else
        echo "Not enough GPU memory available on GPU $GPU_INDEX. Waiting..."
        sleep 60  # Check again in 60 seconds
    fi
done


