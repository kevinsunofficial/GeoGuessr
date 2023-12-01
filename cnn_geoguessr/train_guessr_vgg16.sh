#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr_vgg16.out
#BSUB -eo hpc_logs/cnnguessr_vgg16.err

model="vgg16"
epoch=100
optim="Adam"
lr="1e-3"
batch=16

conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --model $model --augment --epochs $epoch --optimizer $optim --lr $lr --batch_size $batch \
    --out_dir "./results/${model}_b${batch}_${optim}${lr}_${epoch}/" \
    --save_model

conda deactivate