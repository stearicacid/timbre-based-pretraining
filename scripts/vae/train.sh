#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATE_TIME=$(date +"%Y-%m-%d_%H-%M")

python src/vae/train.py 