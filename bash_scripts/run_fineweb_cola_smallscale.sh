#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain.py \
    -- \
    --dit-heads 12 \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --attn-backend sdpa \
    --optimizer muon \
    --run="fineweb_default_cola_smallscale" \
    --device-batch-size 4 \
    --total-batch-size 524288 \
    --target-param-data-ratio 12 \
