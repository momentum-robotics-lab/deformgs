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
        python3 render_experimental.py --model_path "output/dnerf_iso_0.01/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --view_skip 1 --time_skip 1 --log_deform \
        --show_flow --flow_skip 10
    done
done
