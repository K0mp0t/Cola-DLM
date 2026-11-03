#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain_custom_bidirectional_vae.py \
    -- \
    --dit-heads 8 \
    --vae-path ../hf_models/custom_mlp84_fc_vae_fineweb \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --attn-backend sdpa \
    --optimizer adamw \
    --run="fineweb_custom_mlp84_fc_vae_bs16" \
    --device-batch-size 32 \
    --dit-block-size 16 \
    --total-batch-size 524288 \
    --vae-sample-max-std-context 1.0 \
    --vae-sample-max-std-target 1.0 \
    --max-padded-tail-len 0