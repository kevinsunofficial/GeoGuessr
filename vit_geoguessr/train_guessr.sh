#!/bin/bash -l

#BSUB -J vitguessr
#BSUB -oo hpc_logs/vitguessr.out
#BSUB -eo hpc_logs/vitguessr.err

depth=8
numheads=8
epoch=100
optim="Adam"
lr="3e-5"
batch=16

conda activate cuda111_torch

python train.py --input_dir /data/leslie/suny4/geo/world_panorama/ \
    --out_dir "./results/d${depth}h${numheads}_b${batch}_${optim}${lr}_${epoch}/" \
    --depth $depth --num_heads $numheads \
    --optimizer $optim --lr $lr --batch_size $batch --epochs $epoch \
    --save_model

conda deactivate
