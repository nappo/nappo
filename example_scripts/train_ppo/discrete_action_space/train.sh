#!/bin/bash

LOG_DIR=/tmp/atari_ppo
INTER=10000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=100000000000
ENV_ID=PongNoFrameskip-v4
SCHEME=2dacs

###############################################################################

EPS=1e-8
LR=2.5e-4
MODEL=CNN
NUM_PROC=8
NUM_STEPS=128
NUM_MINI_BATCH=4
CLIP_PARAM=0.1
GAMMA=0.99
PPO_EPOCH=3
GAE_LAMBDA=0.95
VALUE_LOSS_COEF=1.0
MAX_GRAD_NORM=0.5
ENTROPY_COEF=0.01
NUM_WORKERS=1
FRAME_STACK=4

###############################################################################

CUDA_VISIBLE_DEVICES=$DEVICES python example_scripts/train_ppo/discrete_action_space/train.py  \
--lr $LR --clip-param $CLIP_PARAM --num-steps $NUM_STEPS --num-mini-batch $NUM_MINI_BATCH \
--num-env-steps $STEPS --log-dir $LOG_DIR --nn $MODEL --gamma $GAMMA --save-interval $INTER \
--ppo-epoch $PPO_EPOCH --gae-lambda $GAE_LAMBDA --use-gae --num-env-processes $NUM_PROC \
--value-loss-coef $VALUE_LOSS_COEF --entropy-coef $ENTROPY_COEF --eps $EPS --max-time $MAX_TIME \
--use_clipped_value_loss --frame-stack $FRAME_STACK --env-id $ENV_ID --max-grad-norm $MAX_GRAD_NORM \
--num-workers $NUM_WORKERS --scheme $SCHEME