#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
# evenly space the isometric lambda values on log scale
export skip_1=2
export skip_2=3
export skip_3=4
export skip_4=5
export skip_5=6
export skip_6=7
export skip_7=8
export skip_20=20
export skip_10=10

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_5="scene_5"
export SCENE_6="scene_6"
export SCENE_7="scene_7"

port=6027
for view_skip in 2 4 8 20 10 ;
#for view_skip in $skip_10 $skip_20;
do
    for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_5 $SCENE_7;
    do 
        python3 train.py -s "data/final_scenes/${SCENE}" --port $port --expname "views_ablation/view_skip_${view_skip}/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project "views_ablation_${SCENE}" --wandb_name "views_skip_${view_skip}" \
        --k_nearest 5 --lambda_isometric 0.01 --view_skip $view_skip 
        # add one to port
        port=$((port+1))
    done
done