#!/bin/bash -l

#BSUB -J vitguessr
#BSUB -oo hpc_logs/vitguessr.out
#BSUB -eo hpc_logs/vitguessr.err

depth=4
numheads=4
epoch=100
optim="Adam"
lr="1e-4"

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --augment --depth $depth --num_heads $numheads --optimizer $optim --lr $lr \
    --epochs $epoch --out_dir "./results/d${depth}h${numheads}_${optim}${lr}_${epoch}/" \
    --save_model

conda deactivate
