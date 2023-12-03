#!/bin/bash -l

#BSUB -J preprocess
#BSUB -oo preprocess.out
#BSUB -eo preprocess.err

conda activate cuda111_torch

python preprocess.py --input_dir /data/leslie/suny4/geo/world_panorama_raw/raw_data/ \
    --out_dir /data/leslie/suny4/geo/geoguessr_data/

conda deactivate
