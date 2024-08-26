export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_4"
export SCENE_5="scene_5"
export SCENE_6="scene_6"

for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6;
do
    python3 render_experimental.py --model_path "output/synthetic/${SCENE}"  --configs arguments/mdnerf-dataset/cube.py --view_skip 5 --time_skip 1 --scale 0.5 --skip_video \
    --show_flow --flow_skip 40 --tracking_window 60 --log_deform 
done