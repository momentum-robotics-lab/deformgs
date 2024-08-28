#!/bin/bash
export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export ISOMETRY=0.3
export LAMBDA_VELOCITY=0.0
export LAMBDA_MOMENTUM=0.03

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_4"
export SCENE_5="scene_5"
export SCENE_6="scene_6"

port=6058 
for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/synthetic/${SCENE}" --port $port --expname "synthetic/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 2000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum $LAMBDA_MOMENTUM --lambda_velocity $LAMBDA_VELOCITY --view_skip 1 --time_skip 1 \
       --checkpoint_iterations 1000 19000 20000  \
       --k_nearest 20 --lambda_isometric $isometry --reg_iter 11000 --staticfying_from 10000 --use_wandb --wandb_project "synthetic_${SCENE}"  --wandb_name "init" 
        port=$((port+1))
    done
done
