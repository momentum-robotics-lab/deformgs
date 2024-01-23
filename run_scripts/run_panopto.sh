export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.3

python3 train.py -s data/panopto/basketball_dnerf/ --port 6028 --expname "panopto/basketball" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING --lambda_momentum 0.1 --k_nearest 5 --lambda_isometric $LAMBDA_ISOMETRIC
