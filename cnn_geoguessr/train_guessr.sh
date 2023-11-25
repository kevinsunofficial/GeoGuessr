#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr.out
#BSUB -eo hpc_logs/cnnguessr.err


conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --model vgg16 --augment --epochs 150 --lr 5e-4 --batch_size 16 \
    --out_dir ./results/baseline_150/ --save_model

conda deactivate
