#!/bin/bash

source ./venv/bin/activate

export BATCH_SIZE=2
export TASKS="mmlu lambada siqa hellaswag"
export MAX_SAMPLES=1000
export DIT_PATH=./cola_pretrain_checkpoints/fineweb_custom_sc_vae/dit
export VAE_PATH=./hf_models/custom_sc_vae_fineweb
export VAE_TYPE="bidirectional"
export OUTPUT_DIR=./eval_output/tasks_custom_sc_vae

NUM_GPUS=$SLURM_GPUS_ON_NODE bash ../scripts/run_benchmark.sh
python ../scripts/calculate_metrics.py $OUTPUT_DIR --tasks $TASKS
