export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

export BASKETBALL="basketball_dnerf"
export JUGGLE="juggle_dnerf"
export SOFTBALL="softball_dnerf"
export FOLD_CLOTH="fold_cloth_short"
export YUNCHAO="real_yunchao_test"

for SCENE in $YUNCHAO; 
do
    python3 render_experimental.py --model_path "output/panopto/${SCENE}"  --configs arguments/mdnerf-dataset/cube.py --view_skip 1 --time_skip 1000 --scale 1.0 --skip_video \
    --show_flow --flow_skip 80 --tracking_window 10  
done