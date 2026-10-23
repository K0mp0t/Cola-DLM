#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain_custom_bidirectional_vae.py \
    -- \
    --dit-heads 8 \
    --vae-path /scratch/users/k25137033/dlms/languageVAE/experiments/10_20_13_46 \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --attn-backend sdpa \
    --optimizer adamw \
    --run="fineweb_custom_fc_vae_IE_limited_std" \
    --device-batch-size 32 \
    --dit-block-size 64 \
    --total-batch-size 524288 \
    --vae-sample-max-std-context 0.25 \
    --vae-sample-max-std-target 0.25 \
    --max-padded-tail-len 0