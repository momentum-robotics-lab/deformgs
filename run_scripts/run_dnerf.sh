#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
# evenly space the isometric lambda values on log scale
export ISOMETRY=0.3

export SCENE_1="bouncingballs"
export SCENE_2="hellwarrior"
export SCENE_3="hook"
export SCENE_4="jumpingjacks"
export SCENE_5="lego"
export SCENE_6="mutant"
export SCENE_7="standup"
export SCENE_8="trex"

port=6027 
for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6 $SCENE_7 $SCENE_8;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/${SCENE}" --port $port --expname "dnerf/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1  \
         --k_nearest 5 --lambda_isometric $isometry \
         --use_wandb --wandb_project "dnerf_${SCENE}" --wandb_name "Init" \
        # add one to port
        port=$((port+1))
    done
done
