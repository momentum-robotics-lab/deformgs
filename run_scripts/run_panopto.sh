export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export LAMBDA_ISOMETRIC=0.3
export LAMBDA_MOMENTUM=0.1

#export RIGIDITY_LAMBDA=0.0
#export LAMBDA_SPRING=0.0
#export LAMBDA_ISOMETRIC=0.0
#export LAMBDA_MOMENTUM=0.0

export DATA_LOCATION="data/panopto/basketball_dnerf"
# export DATA_LOCATION="data/final_scenes/scene_1/"

export BASKETBALL="basketball_dnerf"
export JUGGLE="juggle_dnerf"
export SOFTBALL="softball_dnerf"
export FOLD_CLOTH="fold_cloth_short"
port=6067
for SCENE in $FOLD_CLOTH; 
do
    python3 train.py -s "data/panopto/${SCENE}" --port 6067 --expname "panopto/${SCENE}_reg_coarse_all" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 --lambda_rigidity $RIGIDITY_LAMBDA \
    --lambda_spring $LAMBDA_SPRING --lambda_momentum $LAMBDA_MOMENTUM --k_nearest 3 --lambda_isometric $LAMBDA_ISOMETRIC --view_skip 20 --time_skip 1  --scale 0.25 \
    --no_reg 
    port=$((port+1))
done

#--bounding_box -0.3 0.3 -0.3 0.3 -0.5 0.6  