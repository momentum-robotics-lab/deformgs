#!/bin/bash
export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export ISOMETRY=0.3
export LAMBDA_VELOCITY=0.0
export LAMBDA_MOMENTUM=0.1

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_4"
export SCENE_5="scene_5"
export SCENE_6="scene_6"

port=6027 
for SCENE in $SCENE_1 $SCENE_5 $SCENE_6 $SCENE_4 $SCENE_3 $SCENE_2 ;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/final_scenes_bg/${SCENE}" --port $port --expname "final_scenes_bg_no_cotrack/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum $LAMBDA_MOMENTUM --lambda_velocity $LAMBDA_VELOCITY \
        --use_wandb --wandb_project "paper_final_${SCENE}_bg"  --wandb_name "no_cotrack" --k_nearest 5 --lambda_isometric $isometry --time_skip 1 --coarse_t0 
        # add one to port
        port=$((port+1))
    done
done
