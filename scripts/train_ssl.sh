#!/bin/bash
# Semi-supervised (teacher-student) training followed by test-set evaluation.
# Usage: bash scripts/train_ssl.sh   (edit the variables below first)
set -e

DATA_ROOT=/path/to/data
DATASET=tennis                        # folder under DATA_ROOT with class.txt / train.json / val.json / test.json
FRAMES_DIR=$DATA_ROOT/$DATASET/frames
LABELS_DIR=$DATA_ROOT/$DATASET

MODEL_CHOICE=VIT                      # E2E | ASTRM | VIT
MODEL_CONFIG="models/configs/vit_configs/vit_small_msgsm_[1,3,5].json"

NUM_GPUS=8
BATCH_SIZE=4                          # per GPU
LR=1e-4
EPOCHS=50
CLIP_LEN=100
OUTPUT_DIR=../logs

cd "$(dirname "$0")/../src"

for PERCENT_USE in 0.1 0.2; do
  for SEED in 42 2026 1; do
    LOG_NAME="${MODEL_CHOICE}_${DATASET}_SSL_p${PERCENT_USE}_seed${SEED}"
    echo "==== $LOG_NAME ===="

    torchrun --standalone --nproc_per_node=$NUM_GPUS ssl_train.py \
        --frames_dir "$FRAMES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --model_choice $MODEL_CHOICE \
        --model_config_file "$MODEL_CONFIG" \
        --output_dir $OUTPUT_DIR \
        --logger_name "$LOG_NAME" \
        --percent_use $PERCENT_USE \
        --seed $SEED \
        --clip_len $CLIP_LEN \
        --crop_dim 224 \
        --batch_size $BATCH_SIZE \
        --num_workers 4 \
        --learning_rate $LR \
        --num_epochs $EPOCHS \
        --start_val_epoch 10 \
        --supervise_warmup_epochs 0 \
        --usefp16

    torchrun --standalone --nproc_per_node=1 test.py \
        --output_dir $OUTPUT_DIR \
        --logger_name "$LOG_NAME" \
        --frames_dir "$FRAMES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --model_choice $MODEL_CHOICE \
        --save
  done
done
