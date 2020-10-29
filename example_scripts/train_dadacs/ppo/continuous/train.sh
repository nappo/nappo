#!/bin/bash

WORKING_DIR=$PWD
LOG_DIR=/tmp/trifinger_robot_ppo
INTER=10000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=100000000000

###############################################################################

EPS=1e-8
LR=0.00025
MODEL=MLP
NUM_PROC=20
NUM_STEPS=6000
NUM_MINI_BATCH=40
CLIP_PARAM=0.2
GAMMA=0.99
PPO_EPOCH=4
GAE_LAMBDA=0.95
VALUE_LOSS_COEF=0.5
ENTROPY_COEF=0.01
FRAME_SKIP=3
FRAME_STACK=4
NUM_GWORKERS=1
NUM_CWORKERS=1

###############################################################################

cd $WORKING_DIR

CUDA_VISIBLE_DEVICES=$DEVICES python example_scripts/train_dadacs/ppo/continuous/train.py  \
--lr $LR --clip-param $CLIP_PARAM --num-steps $NUM_STEPS --num-mini-batch $NUM_MINI_BATCH \
--num-env-steps $STEPS --log-dir $LOG_DIR --nn $MODEL --gamma $GAMMA --save-interval $INTER \
--ppo-epoch $PPO_EPOCH --gae-lambda $GAE_LAMBDA --use-gae --num-env-processes $NUM_PROC \
--value-loss-coef $VALUE_LOSS_COEF --entropy-coef $ENTROPY_COEF --eps $EPS --max-time $MAX_TIME \
--use_clipped_value_loss --frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK \
--num-grad-workers $NUM_GWORKERS --num-col-workers $NUM_CWORKERS