# General idea

Latent diffusion with a auxiliary VAE has been a SOTA in image generation for a long time. Recently Bytedance researchers published their Cola-DLM [paper](https://arxiv.org/pdf/2605.06548) and [code](https://github.com/ByteDance-Seed/Cola-DLM) which uses similar method to latent image diffusion but with different goal: VAE in latent image diffusion is used to significantly lower the computational complexity of training and generation while Cola uses VAE to split two text generation tasks into two models. In Cola, DiT's task is to create semantics and VAE's task is to create a token representation of DiT-generated semantics. However, I belive that besides being engineeringly good-looking this idea can be further impoved in a way similar to how and why latent image diffusion uses auxiliary VAE/RAE/other model.

Early experiments have shown that text semantics can be successfully compressed into (and decompressed from) much smaller sequences almost lossless. So I decided to try and train Cola's DiT from scratch with my VAE capable of compressing sequences. I have largely followed Cola's code and methodology so my comparison can be correct from the first experiments.

# Running

There are two different scenarios available right now: Cola-based one which I use to reproduce Cola's metrics on different-scale models and the second one which uses my VAE which has slightly different needs due to sequence compression.

## Prerequisites for running experiments with this code

0. Create a venv however you like to do it
1. ```pip install -r requirements.txt```
2. Download FineWeb 10B tokens sample from [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT)
3. Download pretrained VAE from [here](https://drive.google.com/drive/folders/137FDPxRRwYwzatRmD1sgISuxf9gDcMBL?usp=sharing) for my VAE and [here](https://huggingface.co/ByteDance-Seed/Cola-DLM/tree/main/cola_dlm/cola_vae) for original Cola's VAE

## Pre-defined run scripts

You should change several arguments there: 
1. ```--data-dir``` should be the path to the directory where you put FineWeb-10B parquets
2. ```--run``` should be the desired name of the run (```--run=dummy``` disables wandb)
3. ```--vae-path``` should be the path to the VAE you want to use (Cola-based scripts can only use Cola's VAE, mine can only use mine)

# Experiments I want to run now

1. Smallscale versions of previous experiments: vanilla Cola with ~1.75x smaller DiT and same DiT with my VAE