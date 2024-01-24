# export RIGIDITY_LAMBDA=0.1
# export LAMBDA_SPRING=0.0
# export LAMBDA_ISOMETRIC=0.1
# export LAMBDA_MOMENTUM=0.1

export RIGIDITY_LAMBDA=0.0
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.0
export LAMBDA_MOMENTUM=0.0

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

python3 train.py -s $DATA_LOCATION --port 6067 --expname "panopto/basketball" --configs arguments/panopto/default.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
--lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 1 --time_skip 1 \
--use_wandb --wandb_project panopto_basketball --wandb_name hypernerf_params
