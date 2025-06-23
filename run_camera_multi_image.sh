#!/bin/bash

# Base path components
DATA_BASE="data/neuman/lab_"
MODEL_BASE="instantsplat/lab_"
LOG_BASE="instantsplat/lab_"

# Loop over a range of indices (you can adjust {0..99})
for i in {20..20}; do
    # Pad with zeros to 5 digits
    #idx=$(printf "%05d" $i)
    idx=$i
    # Construct paths
    DATA_PATH="${DATA_BASE}${idx}_all"
    MODEL_PATH="${MODEL_BASE}${idx}_all"
    LOG_PATH="${LOG_BASE}${idx}_all/2_train.log"

    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_PATH")"

    # Construct and run the command
    echo "Running training for index $idx"
    CUDA_VISIBLE_DEVICES=0 python vggt/train_instantsplat.py \
        -s "$DATA_PATH" \
        -m "$MODEL_PATH" \
        -r 1 \
        --n_views $i \
        --iterations 1 \
        --pp_optimizer \
        --optim_pose \
        > "$LOG_PATH" 2>&1
done
