export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01

# python3 train.py -s data/final_scenes/scene_5/ --port 6027 --expname "time_ablation/full" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project time_ablation --wandb_name full --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
python3 render.py --model_path "output/time_ablation/full/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 200 --log_deform 

# python3 train.py -s data/final_scenes/scene_5/ --port 6028 --expname "time_ablation/skip_2" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project time_ablation --wandb_name skip_2 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --time_skip 2
# python3 render.py --model_path "output/time_ablation/skip_2/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --log_deform 

# python3 train.py -s data/final_scenes/scene_5/ --port 6029 --expname "time_ablation/skip_3" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project time_ablation --wandb_name skip_3 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --time_skip 3
# python3 render.py --model_path "output/time_ablation/skip_3/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --log_deform 

# python3 train.py -s data/final_scenes/scene_5/ --port 6030 --expname "time_ablation/skip_4" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project time_ablation --wandb_name skip_4 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --time_skip 4
# python3 render.py --model_path "output/time_ablation/skip_4/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --log_deform 

# python3 train.py -s data/final_scenes/scene_5/ --port 6031 --expname "time_ablation/skip_5" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project time_ablation --wandb_name skip_5 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC --time_skip 5
# python3 render.py --model_path "output/time_ablation/skip_5/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 2 --log_deform 

# python3 metrics.py -m output/time_ablation/*