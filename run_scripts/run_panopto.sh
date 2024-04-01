export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01
export LAMBDA_MOMENTUM=0.1

#export RIGIDITY_LAMBDA=0.0
#export LAMBDA_SPRING=0.0
#export LAMBDA_ISOMETRIC=0.0
#export LAMBDA_MOMENTUM=0.0

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

export BASKETBALL="basketball_dnerf"
export JUGGLE="juggle_dnerf"
export SOFTBALL="softball_dnerf"
export FOLD_CLOTH="fold_cloth_short"
port=6067
for SCENE in $FOLD_CLOTH; 
do
    python3 train.py -s "data/panopto/${SCENE}" --port 6067 --expname "panopto/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
    --lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 2000 --time_skip 1 --view_skip 4 --no_reg --coarse_t0 --scale 0.25
    port=$((port+1))
done
#python3 train.py -s $DATA_LOCATION --port 6067 --expname "panopto/basketball_all_momentum" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
#--lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 1 --time_skip 1 --no_coarse  \
#--use_wandb --wandb_project panopto_basketball --wandb_name pointcloud_init_all_ts_with_reg_150_grid