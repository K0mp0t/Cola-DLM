#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:33999 \
    ../scripts/cola_pretrain.py \
    -- \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --attn-backend sdpa \
    --optimizer adamw \
    --dit-heads 8 \
    --run="fineweb_default_cola_04b" \
    --device-batch-size 16 \
    --total-batch-size 524288 \
    --target-param-data-ratio 12 

