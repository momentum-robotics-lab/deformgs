export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01
export N_FLOW_TRAJS=200
export VIEW_SKIP=1
export TIME_SKIP=1


python3 render_experimental.py --model_path "output/dyna_static/scene_5_all_ts_coarse_bigger_cube_init" --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 100 --time_skip 5 --iteration 20000 --load_coarse --load_checkpoint 
