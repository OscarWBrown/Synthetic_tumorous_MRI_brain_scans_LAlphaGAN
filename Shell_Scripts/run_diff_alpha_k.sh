#!/bin/bash

alpha_ds=(0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2 4) #3
alpha_gs=(0.75 1 1.5 2 3 4 8 16) # 0.25 0.50 32 64
# ks=(0.1 0.25 0.75 1.0 1.25 1.5 1.75 2.0)

for alpha_d in "${alpha_ds[@]}"
do
    for alpha_g in "${alpha_gs[@]}"
    do
        # for k in "${ks[@]}"
        # do
            echo "Running experiment with alpha_d=${alpha_d} and alpha_g=${alpha_g}" #and k=${k}"
            python MRIALPHA128_diff_alpha.py --alpha_d "${alpha_d}" --alpha_g "${alpha_g}" #--k ${ks}
        # done  
    done
done

