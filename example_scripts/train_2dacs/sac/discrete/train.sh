#!/bin/bash

WORKING_DIR=$PWD
LOG_DIR=/tmp/atari_sac
INTER=50000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=10000000000000
ENV_ID=MsPacmanNoFrameskip-v4

###############################################################################

LR=0.0003
MODEL=CNN
NUM_PROC=1
GAMMA=0.99
POLYAK=0.995
MINI_BATCH_SIZE=64
UPDATE_EVERY=4
NUM_UPDATES=1
START_STEPS=20000
BUFFER_SIZE=300000
FRAME_SKIP=4
FRAME_STACK=4
TARGET_UPDATE_INTERVAL=8000
NUM_WORKERS=1

###############################################################################

cd $WORKING_DIR

CUDA_VISIBLE_DEVICES=$DEVICES python example_scripts/train_2dacs/sac/discrete/train.py  \
--num-env-steps $STEPS --log-dir $LOG_DIR --gamma $GAMMA --polyak $POLYAK \
--save-interval $INTER --max-time $MAX_TIME --num-env-processes $NUM_PROC \
--mini-batch-size $MINI_BATCH_SIZE --update-every $UPDATE_EVERY --num-updates $NUM_UPDATES \
--start-steps $START_STEPS --buffer-size $BUFFER_SIZE --lr $LR --nn $MODEL \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --env-id $ENV_ID \
--target-update-interval $TARGET_UPDATE_INTERVAL --num-workers $NUM_WORKERS
