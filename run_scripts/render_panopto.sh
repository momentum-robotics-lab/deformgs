export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

export BASKETBALL="basketball_dnerf"
export JUGGLE="juggle_dnerf"
export SOFTBALL="softball_dnerf"
export FOLD_CLOTH="fold_cloth_short"


for SCENE in $FOLD_CLOTH; 
do
    python3 render_experimental.py --model_path "output/panopto/${SCENE}_reg_coarse_all"  --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --time_skip 1 --scale 0.25 --skip_video \
    --show_flow --flow_skip 80 --tracking_window 10  
done