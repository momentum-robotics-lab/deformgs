export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=1.0
export LAMBDA_MOMENTUM=0.5

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
export YUNCHAO="real_yunchao_test"
port=6080

for SCENE in $YUNCHAO; 
do
    python3 train.py -s "data/panopto/${SCENE}" --port 6067 --expname "panopto/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
    --lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 1 --time_skip 1  --scale 0.5 \
    --mask_loss_from 0 --checkpoint_iterations 0 100 10000 15000 20000 30000 --reg_iter 10000 --lambda_cotrack 0.3 \
    --use_wandb --wandb_project "panopto" --wandb_name "${SCENE}_pyramid" --cotrack_loss_from 50000 --coarse_t0 
    port=$((port+1))
done

#--bounding_box -0.3 0.3 -0.3 0.3 -0.5 0.6  