#!/bin/bash

alphas=(128 256 512 1024 2048)

# Unset CUDA-related environment variables
unset CUDA_VISIBLE_DEVICES
unset NVIDIA_VISIBLE_DEVICES
for alpha in "${alphas[@]}"
do
    echo "Running experiment with alpha_d=${alpha} and alpha_g=${alpha}"
    python MRIALPHA64_Alpha_k_variable"${alpha}".py
done

# tmux new -s oscarsession
# ./cpu...
# control + b then d
# tmux attach -t oscarsession
# tmux kill-session

# keep d ~ 1 and alpha vary 
# check out LSkGAN for values of k
# check out justins experiment section