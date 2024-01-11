export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

python3 train.py -s data/final_scene_5 --port 6021 --expname "final_scenes/scene_2" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 50 --time_skip 5
