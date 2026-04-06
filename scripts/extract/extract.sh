export TF_FORCE_GPU_ALLOW_GROWTH=true

now=$(date +"%Y%m%d_%H%M%S")

# Hydra key=value overrides.
PYTHONPATH=src python -m extract.main 