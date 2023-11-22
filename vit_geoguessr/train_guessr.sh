#!/bin/bash -l

#BSUB -J vitguessr
#BSUB -oo hpc_logs/vitguessr.out
#BSUB -eo hpc_logs/vitguessr.err

depth=8
numheads=8
epoch=100
optim="Adam"

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --depth $depth --num_heads $numheads --lr 1e-4 --optimizer $optim \
    --epochs $epoch --out_dir "./results/d${depth}h${numheads}_${epoch}/" \
    --save_model

conda deactivate
