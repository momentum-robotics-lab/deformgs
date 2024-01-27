#!/bin/bash
export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export isometry=0.316227766

port=6027 

python3 render_experimental.py --model_path "output/mimic/mimic_1_2_demonstrations" --skip_video \
        --configs arguments/mdnerf-dataset/cube.py --view_skip 1000  --time_skip 9 --log_deform --show_flow --flow_skip 10
 
