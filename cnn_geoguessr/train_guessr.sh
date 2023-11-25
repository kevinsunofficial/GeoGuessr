#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr.out
#BSUB -eo hpc_logs/cnnguessr.err

model="vgg16"
epoch=150
optim="Adam"
lr="5e-4"
batch=16

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --model $model --augment --epochs $epoch \
    --optimizer $optim --lr 5e-4 --batch_size $batch \
    --out_dir "./results/${model}_b${batch}_${optim}${lr}_${epoch}/" \
    --save_model

conda deactivate
