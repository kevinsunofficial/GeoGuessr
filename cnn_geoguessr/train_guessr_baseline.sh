#!/bin/bash -l

#BSUB -J cnnguessr_baseline
#BSUB -oo hpc_logs/cnnguessr_baseline.out
#BSUB -eo hpc_logs/cnnguessr_baseline.err

model="baseline"
epoch=100
optim="Adam"
lr="1e-3"
batch=16

conda activate cuda111_torch

python train.py --input_dir /data/leslie/suny4/geo/geoguessr_data/ \
    --out_dir "./results/${model}_b${batch}_${optim}${lr}_${epoch}/" \
    --model $model --epochs $epoch --optimizer $optim --lr $lr --batch_size $batch \
    --save_model

conda deactivate
