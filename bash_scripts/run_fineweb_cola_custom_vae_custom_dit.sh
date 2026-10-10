#!/bin/bash

source ./venv/bin/activate

torchrun \
    --nnodes=1 \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    ../scripts/cola_pretrain_custom_bidirectional_vae.py \
    -- \
    --dit-heads 12 \
    --vae-path ./hf_models/custom_sc_vae_fineweb \
    --data-dir /scratch/users/k25137033/dlms/languageVAE/data/fineweb \
    --optimizer muon \
    --run="fineweb_custom_sc_vae_custom_dit" \
    --device-batch-size 16 \
    --total-batch-size 500000 \
    --target-param-data-ratio 12
