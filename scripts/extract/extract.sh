export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

now=$(date +"%Y%m%d_%H%M%S")

# Hydra key=value overrides.
PYTHONPATH=src taskset -c 0-79 python -m extract.main \
    extraction.split=train \
    dataset.max_samples=10 \
    runtime.workers_per_gpu=8 \
    extraction.save_features=true \
    paths.output_root=outputs/harmonic_${now} \
    dataset.data_dir="/mlnas/rin/tensorflow_datasets"