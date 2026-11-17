#!/bin/bash
# Supervised baseline training followed by test-set evaluation.
# Usage: bash scripts/train_sl.sh   (edit the variables below first)
set -e

DATA_ROOT=/path/to/data
DATASET=tennis                        # folder under DATA_ROOT with class.txt / train.json / val.json / test.json
FRAMES_DIR=$DATA_ROOT/$DATASET/frames
LABELS_DIR=$DATA_ROOT/$DATASET

MODEL_CHOICE=E2E                      # E2E | ASTRM | VIT
MODEL_CONFIG="models/configs/e2e_configs/e2e_gsm.json"

NUM_GPUS=2
BATCH_SIZE=4                          # per GPU
LR=1e-3
EPOCHS=50
CLIP_LEN=100
OUTPUT_DIR=../logs

cd "$(dirname "$0")/../src"

for PERCENT_USE in 0.1 0.2 1.0; do
  for SEED in 42 2026 1; do
    LOG_NAME="${MODEL_CHOICE}_${DATASET}_SL_p${PERCENT_USE}_seed${SEED}"
    echo "==== $LOG_NAME ===="

    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train.py \
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
        --num_workers 8 \
        --learning_rate $LR \
        --num_epochs $EPOCHS \
        --start_val_epoch 10 \
        --usefp16

    torchrun --standalone --nproc_per_node=$NUM_GPUS test.py \
        --output_dir $OUTPUT_DIR \
        --logger_name "$LOG_NAME" \
        --frames_dir "$FRAMES_DIR" \
        --labels_dir "$LABELS_DIR" \
        --model_choice $MODEL_CHOICE \
        --save
  done
done
