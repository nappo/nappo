#!/bin/bash

LOG_DIR=/tmp/pybullet_sac
INTER=50000
DEVICES="0,1,2,3"
MAX_TIME=100000000000000000
STEPS=100000000000000000
ENV_ID=HalfCheetahBulletEnv-v0

###############################################################################

LR=3e-4
MODEL=MLP
NUM_PROC=1
GAMMA=0.99
POLYAK=0.995
MINI_BATCH_SIZE=256
UPDATE_EVERY=1
NUM_UPDATES=1
START_STEPS=20000
BUFFER_SIZE=1000000
FRAME_SKIP=0
FRAME_STACK=1
NUM_WORKERS=1

###############################################################################

CUDA_VISIBLE_DEVICES=$DEVICES python python_scripts/train_pybullet/sac/train.py  \
--num-env-steps $STEPS --log-dir $LOG_DIR --gamma $GAMMA --polyak $POLYAK \
--save-interval $INTER --max-time $MAX_TIME --num-env-processes $NUM_PROC \
--mini-batch-size $MINI_BATCH_SIZE --update-every $UPDATE_EVERY --num-updates $NUM_UPDATES \
--start-steps $START_STEPS --buffer-size $BUFFER_SIZE --lr $LR --nn $MODEL \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --num-workers $NUM_WORKERS \
--env-id $ENV_ID
