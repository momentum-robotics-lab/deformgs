export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

python3 train.py -s $DATA_LOCATION --port 6021 --expname "panopto/basketball" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
--lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 1 --time_skip 1000 \
--use_wandb --wandb_project panopto_basketball --wandb_name no_deformation_t0_only
