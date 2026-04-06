NOW="$(date +"%Y-%m-%d_%H-%M-%S")"
OUTPUT_DIR="outputs_$NOW"
K="${1:-39}"

export CUDA_VISIBLE_DEVICES=3

python src/inference/run.py \
  clustering.k=$K \

