#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr.out
#BSUB -eo hpc_logs/cnnguessr.err

model="baseline"
epoch=100
optim="Adam"
lr="1e-3"
batch=32

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --model $model --epochs $epoch --optimizer $optim --lr $lr --batch_size $batch \
    --out_dir "./results/${model}_b${batch}_${optim}${lr}_${epoch}/" \
    --save_model

conda deactivate
