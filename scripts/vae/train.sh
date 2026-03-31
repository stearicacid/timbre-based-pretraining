#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATE_TIME=$(date +"%Y-%m-%d_%H-%M")

taskset -c 0-47 python src/vae/train.py \
    experiment_name=vae