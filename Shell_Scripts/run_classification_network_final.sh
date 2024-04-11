#!/bin/bash

path1='cleaned128by128augmented_d1_g5_tumor'
path2='cleaned128by128augmented_d1_g5_less_both'
path3='cleaned128by128'
lrs=(0.05 0.005 0.0005)
batch_sizes=(64)

# Required free memory in MiB
REQUIRED_FREE_MEMORY=20000  # Adjust this to your needs

# GPU index to check (1 for the second GPU)
GPU_INDEX=1

# Function to check available GPU memory on a specific GPU
check_gpu_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_INDEX | awk '{print $1}'
}

while true; do
    AVAILABLE_MEMORY=$(check_gpu_memory)  # Capture the output of check_gpu_memory function

    echo "Available GPU memory on GPU $GPU_INDEX: $AVAILABLE_MEMORY MiB"

    if [ "$AVAILABLE_MEMORY" -ge "$REQUIRED_FREE_MEMORY" ]; then
        echo "Enough GPU memory available on GPU $GPU_INDEX. Starting training."
        # Loop through learning rates and batch sizes
        for lr in "${lrs[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                echo "Running experiment with lr=${lr} and batch_size=${batch_size} and train_path=${path2}"
                python classification_network_pytorch.py --lr ${lr} --batch_size ${batch_size} --train_path "${path2}" --num 2
                # echo "Running experiment with lr=${lr} and batch_size=${batch_size} and train_path=${path3}"
                # python classification_network_pytorch.py --lr ${lr} --batch_size ${batch_size} --train_path "${path3}" --num 2
            done
        done
        # for i in {2..6}
        # do
        #     python classification_network_pytorch.py --lr 0.05 --batch_size 128 --train_path "${path1}" --num ${i}
        #     python classification_network_pytorch.py --lr 0.05 --batch_size 128 --train_path "${path2}" --num ${i}
        #     python classification_network_pytorch.py --lr 0.05 --batch_size 128 --train_path "${path3}" --num ${i}
        # done
        # break
    else
        echo "Not enough GPU memory available on GPU $GPU_INDEX. Waiting..."
        sleep 15  # Check again in 15 seconds
    fi
done
