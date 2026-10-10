#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain_custom_bidirectional_vae.py \
    -- \
    --dit-heads 12 \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --vae-path ./hf_models/custom_sc_vae_fineweb \
    --attn-backend sdpa \
    --optimizer muon \
    --run="fineweb_cola_smallscale_context_independent_encoding" \
    --device-batch-size 4 \
    --total-batch-size 500000 \
    --dit-block-size 16 \
    --max-seq-len 1024 \
    --target-param-data-ratio 12
