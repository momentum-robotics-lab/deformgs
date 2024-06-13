export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.3
export LAMBDA_MOMENTUM=0.03

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
    python3 train.py -s "data/panopto/${SCENE}" --port 6067 --expname "panopto/${SCENE}_reg_coarse_all_more_iso" --configs arguments/mdnerf-dataset/cube.py --lambda_w 2000 --lambda_rigidity $RIGIDITY_LAMBDA \
    --lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 20 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 4 --time_skip 1  --scale 0.25 \
    --no_reg --use_wandb --wandb_project "panopto" --wandb_name "${SCENE}_20_views" --mask_loss_from 2000 --checkpoint_iterations 100 1000 5000 10000 15000 20000 
    port=$((port+1))
done
