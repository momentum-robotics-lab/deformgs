#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export ISOMETRY=0.01

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_4"
export SCENE_5="scene_5"
export SCENE_6="scene_6"

port=6027 
#for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6;
for SCENE in $SCENE_4  ;
do
    for isometry in $ISOMETRY;
    do 

        python3 render_experimental.py --model_path "output/final_scenes_bg_20m_range_exp_iso/${SCENE}_iso_static_simple_15" --configs arguments/mdnerf-dataset/cube.py --skip_train --skip_video --view_skip 200 --time_skip 1  \
        --show_flow --flow_skip 1 --no_gt --tracking_window 100 --scale 1 
        # add one to port
        port=$((port+1))
    done
done
