#!/bin/bash
reductions=("shuffle")
factors=(8)
pes=("lape")
lrs=(0.001)

for red in "${reductions[@]}"
do
    for fac in "${factors[@]}"
    do
        for pe in "${pes[@]}"
        do
            for lr in "${lrs[@]}"
            do
                sbatch --partition=main pretrain.sh "$red" "$fac" "$pe" "$lr"
            done
        done
    done
done