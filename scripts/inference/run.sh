NOW="$(date +"%Y-%m-%d_%H-%M-%S")"
OUTPUT_DIR="/mlnas/rin/ours/wotriplet_$NOW"
K="${1:-39}"

export CUDA_VISIBLE_DEVICES=3

taskset -c 49-68 python src/inference/run.py \
  clustering.k=$K \
  output.root_dir=$OUTPUT_DIR \
  model.checkpoint_path="/home/rin/timbre-based-pretraining/outputs/triplet_loss/2025-08-21_22-13-30/best_model.pth" \
  model.train_config_path="/home/rin/timbre-based-pretraining/outputs/triplet_loss/2025-08-21_22-13-30/.hydra/config.yaml" \
  data.split_info_pkl="/mlnas/rin/datasets/sato2024_main_loader/split_info.pkl" \
  data.lakh_dataset_dir="/mlnas/rin/datasets/lmd/lmd_full" \
  data.dismiss_midis="/mlnas/rin/datasets/sato2024_main_loader/dead.txt" \
  data.normalizer_path="/home/rin/timbre-based-pretraining/outputs/normalizer_harmonic.json" \
  data.train_data_dir="/data3/rin/ddsp/train" \
  data.valid_data_dir="/data3/rin/ddsp/valid" \
  data.test_data_dir="/data3/rin/ddsp/test" \
  synthesis.max_midis=5
