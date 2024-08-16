export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

export BASKETBALL="basketball_dnerf"
export JUGGLE="juggle_dnerf"
export SOFTBALL="softball_dnerf"
export FOLD_CLOTH="fold_cloth_short"
export CLOTH_FILTERED="cloth_filtered"

for SCENE in $FOLD_CLOTH; 
do
    python3 render_experimental.py --model_path "output/panopto/${SCENE}_reg_coarse_all"  --configs arguments/mdnerf-dataset/cube.py --view_skip 5 --time_skip 1 --scale 0.5 --skip_video \
    --show_flow --flow_skip 40 --tracking_window 60 --log_deform 
done