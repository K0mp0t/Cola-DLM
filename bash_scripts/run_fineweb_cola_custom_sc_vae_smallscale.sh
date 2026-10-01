#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain_custom_vae.py \
    -- \
    --dit-heads 12 \
    --vae-path ./hf_models/custom_sc_vae_fineweb \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --attn-backend sdpa \
    --optimizer muon \
    --run="fineweb_custom_sc_vae_small_blocksize" \
    --device-batch-size 4 \
    --total-batch-size 524288 \
    --target-param-data-ratio 12 \
    --dit-block-size 2
