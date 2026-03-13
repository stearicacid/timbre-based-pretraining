#!/usr/bin/env bash
set -euo pipefail

cd environments/dev
docker compose build timbre_based_pretraining
docker compose up timbre_based_pretraining -d

docker compose exec timbre_based_pretraining bash -lc '
  set -euo pipefail
  uv python install 3.8.10
  uv python pin 3.8.10
  uv venv
  uv pip install "setuptools<82" wheel
  uv pip install --no-build-isolation crepe==0.0.12
  uv pip install -r requirements.txt
  uv pip install ddsp==3.7.0
  uv pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
'

docker compose exec timbre_based_pretraining bash
source .venv/bin/activate