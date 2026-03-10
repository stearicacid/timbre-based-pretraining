#!/usr/bin/env bash

cd environments/gpu
docker compose build timbre_based_pretraining
docker compose up timbre_based_pretraining -d
docker compose exec timbre_based_pretraining bash

uv python install 3.9
uv python pin 3.9

uv venv
uv add -r requirements.txt
uv pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118

source .venv/bin/activate
