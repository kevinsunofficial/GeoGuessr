#!/bin/bash -l

#BSUB -J cnnguessr_vgg
#BSUB -oo hpc_logs/cnnguessr_vgg.out
#BSUB -eo hpc_logs/cnnguessr_vgg.err

model="vgg"
epoch=250
optim="Adam"
lr="5e-4"
batch=16

conda activate cuda111_torch

python train.py --input_dir /data/leslie/suny4/geo/geoguessr_data/ \
    --out_dir "./results/${model}_b${batch}_${optim}${lr}_${epoch}/" \
    --model $model --epochs $epoch --optimizer $optim --lr $lr --batch_size $batch \
    --save_model

conda deactivate
