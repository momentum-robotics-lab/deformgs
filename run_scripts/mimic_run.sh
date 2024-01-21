#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export isometry=0.316227766

port=6027 

# python3 train.py -s "data/mimic/mimic_1" --port $port --expname "mimic/mimic_1" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
# --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 \
# --k_nearest 5 --lambda_isometric $isometry --time_skip 4 \
# --use_wandb --wandb_project "mimic_1" --wandb_name "First_try" \
# --use_wandb --wandb_project "time_interpolation_${SCENE}" --wandb_name "t_skip_4" \

python3 train.py -s "data/mimic/mimic_1" --port $port --expname "mimic/mimic_1_no_reg" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
--lambda_rigidity 0.0 --lambda_spring 0.0  --lambda_momentum 0.0 \
--k_nearest 5 --lambda_isometric 0.0 --time_skip 1 \
--use_wandb --wandb_project "mimic_1" --wandb_name "no_reg_terms" \
 
