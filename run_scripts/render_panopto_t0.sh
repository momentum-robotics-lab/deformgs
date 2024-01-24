export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.1

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

python3 render_experimental.py --model_path "output/panopto/basketball"  --configs arguments/mdnerf-dataset/cube.py --view_skip 1 --time_skip 10000 