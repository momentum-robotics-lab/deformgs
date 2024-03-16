#!/bin/bash
export TIME_SKIP=4

export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export ISOMETRY=0.3

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_5="scene_5"
export SCENE_6="scene_6"
export SCENE_7="scene_7"

port=6027 
for SCENE in $SCENE_1 $SCENE_2 $SCENE_3;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/final_scenes/${SCENE}" --port $port --expname "time_interpolation/${SCENE}_${isometry}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project "time_interpolation_${SCENE}" --wandb_name "iso_$isometry" --k_nearest 5 --lambda_isometric $isometry \
        --time_skip $TIME_SKIP
        # add one to port
        port=$((port+1))
    done
done