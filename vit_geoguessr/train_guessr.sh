#!/bin/bash -l

#BSUB -J vitguessr
#BSUB -oo hpc_logs/vitguessr.out
#BSUB -eo hpc_logs/vitguessr.err

depth=8
numheads=8
epoch=50
optim="Adam"
lr="5e-5"
batch=16
augment="aug"

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --augment --depth $depth --num_heads $numheads --optimizer $optim --lr $lr --batch_size $batch \
    --epochs $epoch --out_dir "./results/${augment}_d${depth}h${numheads}_b${batch}_${optim}${lr}_${epoch}/" \
    --save_model

conda deactivate
