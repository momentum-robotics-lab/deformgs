export XARM_FOLD="xarm_fold_new_mask"
export FOLD_CLOTH=""
export CLOTH="cloth"

export DUVET="duvet"
export CLOTH="cloth"
export XARM_FOLD="xarm_fold_tshirt"

for SCENE in $XARM_FOLD $DUVET $CLOTH; 
do
    python3 render_experimental.py --model_path "output/robo360/${SCENE}"  --configs arguments/mdnerf-dataset/cube.py --view_skip 5 --time_skip 1 --scale 0.5 --skip_video \
    --show_flow --flow_skip 40 --tracking_window 60 --log_deform 
done