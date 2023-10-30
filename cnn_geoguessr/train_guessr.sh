#!/bin/bash -l

#BSUB -J cnnguessr
#BSUB -oo hpc_logs/cnnguessr.out
#BSUB -eo hpc_logs/cnnguessr.err


conda activate cuda111_torch

python cnn_geoguessr/train.py --root_dir /data/leslie/suny4/geo/world_panorama/ \
    --epochs 150 --save_model ./results/ --plot_loss ./results/ --plot_map ./results/

conda deactivate
