export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01
export N_FLOW_TRAJS=200
export VIEW_SKIP=1
export TIME_SKIP=1

python3 train.py -s data/final_scenes_bg/scene_5/ --port 6021 --expname "dyna_static/scene_5_all_ts_coarse_bigger_cube_init" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 \
--k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 1 --time_skip 1 --reg_iter 500000 
#--use_wandb --wandb_project dyna-static --wandb_name coarse_t0_only_main
