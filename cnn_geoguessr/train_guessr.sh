#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr.out
#BSUB -eo hpc_logs/cnnguessr.err


conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --epochs 100 --out_dir ./results/baseline_100/ --save_model

conda deactivate
