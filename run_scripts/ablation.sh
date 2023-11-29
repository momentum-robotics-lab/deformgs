export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.01

# python3 train.py -s data/final_scenes/scene_1/ --port 6020 --expname "final_scenes/scene_1" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 --use_wandb --wandb_project final_scene_1 --wandb_name big_exp --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
# python3 render.py --model_path "output/final_scenes/scene_1/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --flow_skip 100 --show_flow --log_deform 
# python3 train.py -s data/final_scenes/scene_2/ --port 6021 --expname "final_scenes/scene_2" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 --use_wandb --wandb_project final_scene_2 --wandb_name big_exp --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
# python3 render.py --model_path "output/final_scenes/scene_2/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --flow_skip 100 --show_flow --log_deform 
# python3 train.py -s data/final_scenes/scene_3/ --port 6023 --expname "final_scenes/scene_3" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project final_scene_3 --wandb_name big_exp --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
# python3 render.py --model_path "output/final_scenes/scene_3/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --flow_skip 100 --show_flow --log_deform 
# python3 train.py -s data/final_scenes/scene_5/ --port 6024 --expname "final_scenes/scene_5" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project final_scene_5 --wandb_name big_exp --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
# python3 render.py --model_path "output/final_scenes/scene_5/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --flow_skip 100 --show_flow --log_deform 
# python3 train.py -s data/final_scenes/scene_6/ --port 6026 --expname "final_scenes/scene_6" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project final_scene_6 --wandb_name big_exp --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
# python3 render.py --model_path "output/final_scenes/scene_6/" --skip_train --skip_video --configs arguments/mdnerf-dataset/cube.py --view_skip 20 --flow_skip 100 --show_flow --log_deform 

python3 train.py -s data/final_scenes/scene_7/ --port 6027 --expname "ablation/full" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project ablation_5 --wandb_name full --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
python3 render_experimental.py --model_path output/ablation/full/ --skip_train --skip_test --configs arguments/mdnerf-dataset/cube.py --view_skip 100 --flow_skip 5 --show_flow

python3 train.py -s data/final_scenes/scene_7/ --port 6028 --expname "ablation/no_rigid" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project ablation_5 --wandb_name no_rigid --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
python3 render_experimental.py --model_path output/ablation/no_rigid/ --skip_train --skip_test --configs arguments/mdnerf-dataset/cube.py --view_skip 100 --flow_skip 5 --show_flow

python3 train.py -s data/final_scenes/scene_7/ --port 6029 --expname "ablation/no_momentum" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.0 --use_wandb --wandb_project ablation_5  --wandb_name no_momentum --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
python3 render_experimental.py --model_path output/ablation/no_momentum/ --skip_train --skip_test --configs arguments/mdnerf-dataset/cube.py --view_skip 100 --flow_skip 5 --show_flow

python3 train.py -s data/final_scenes/scene_7/ --port 6030 --expname "ablation/no_iso" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project ablation_5 --wandb_name no_iso --k_nearest 5 --lambda_isometric 0.0
python3 render_experimental.py --model_path output/ablation/no_iso/ --skip_train --skip_test --configs arguments/mdnerf-dataset/cube.py --view_skip 100 --flow_skip 5 --show_flow