#!/usr/bin/env bash
MODEL_TYPE="mlp"
# MODEL_NAME_OR_PATH="distilbert-base-uncased"
TOKENIZER_NAME="bert-base-uncased"
BATCH_SIZE=16
EPOCHS=100

set -e # stop if anything goes wrong 

WEIGHT_MEASURE="AbsMean"
RESULTS_FILE="results_mlp_bitlinear-$WEIGHT_MEASURE.csv"

for DATASET in "20ng" "r8" "r52" "ohsumed" "mr"; do
        python3 run.py --model_type "$MODEL_TYPE" --tokenizer_name "$TOKENIZER_NAME" \
                --batch_size $BATCH_SIZE --learning_rate "0.001"\
                --epochs $EPOCHS --num_workers 1 --bitlinear --bitlinear_weight_measure $WEIGHT_MEASURE --results_file "$RESULTS_FILE" "$DATASET"
done

