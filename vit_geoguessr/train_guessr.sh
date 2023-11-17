#!/bin/bash -l

#BSUB -J vitguessr
#BSUB -oo hpc_logs/vitguessr.out
#BSUB -eo hpc_logs/vitguessr.err


conda activate cuda111_torch

python train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --epochs 50 --out_dir ./results/ --save_model

conda deactivate
