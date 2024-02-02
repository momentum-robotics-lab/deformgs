export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export ISOMETRY=0.01

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_7"
export SCENE_5="scene_5"
export SCENE_6="scene_6"
export SCENE_7="scene_7"

port=6027 
for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6;
do
    for isometry in $ISOMETRY;
    do 
     python3 render_experimental.py --model_path "output/shadow_ablation/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --view_skip 10 --time_skip 1 --log_deform \
        --show_flow --flow_skip 10 --no_shadow
    done
done
