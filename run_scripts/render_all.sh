export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01

python3 render_experimental.py --model_path "output/final_scenes/scene_1/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform 
python3 render_experimental.py --model_path "output/final_scenes/scene_2/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform 
python3 render_experimental.py --model_path "output/final_scenes/scene_3/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform +
python3 render_experimental.py --model_path "output/final_scenes/scene_4/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform 
python3 render_experimental.py --model_path "output/final_scenes/scene_5/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform 
python3 render_experimental.py --model_path "output/final_scenes/scene_6/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --time_skip 2 --log_deform 