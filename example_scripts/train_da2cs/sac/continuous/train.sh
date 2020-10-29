#!/bin/bash

WORKING_DIR=$PWD
LOG_DIR=/tmp/trifinger_robot_sac
INTER=50000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=100000000000

###############################################################################

LR=0.00025
MODEL=MLP
NUM_PROC=1
GAMMA=0.99
POLYAK=0.995
MINI_BATCH_SIZE=64
UPDATE_EVERY=1
NUM_UPDATES=1
START_STEPS=1250
BUFFER_SIZE=500000
FRAME_SKIP=3
FRAME_STACK=0
NUM_WORKERS=4

###############################################################################

cd $WORKING_DIR

CUDA_VISIBLE_DEVICES=$DEVICES python example_scripts/train_da2cs/sac/continuous/train.py  \
--num-env-steps $STEPS --log-dir $LOG_DIR --gamma $GAMMA --polyak $POLYAK \
--save-interval $INTER --max-time $MAX_TIME --num-env-processes $NUM_PROC \
--mini-batch-size $MINI_BATCH_SIZE --update-every $UPDATE_EVERY --num-updates $NUM_UPDATES \
--start-steps $START_STEPS --buffer-size $BUFFER_SIZE --lr $LR --nn $MODEL \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --num-workers $NUM_WORKERS