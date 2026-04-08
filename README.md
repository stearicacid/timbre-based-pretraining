# Timbre-Based Pretraining

This repository contains the upstream pipeline of our ICASSP 2026 paper:

**Timbre-Based Pretraining with Pseudo-Labels for Multi-Instrument Automatic Music Transcription**

Specifically, this release covers:

1. timbre feature extraction from NSynth using DDSP-derived harmonic controls,
2. VAE training for latent timbre-space learning,
3. k-means-based pseudo-label generation and MIDI-conditioned audio synthesis.

## Scope of this repository

This is a **partial release of the full paper pipeline**.

Included in this repository:
- timbre representation learning,
- pseudo-label construction,
- synthetic audio generation and packaging for pretraining.

Not included in this repository:
- downstream AMT model pretraining,
- fine-tuning on instrument-labeled datasets,
- transcription evaluation.

In the paper, the downstream AMT backbone is **Jointist**.
Please refer to the Jointist implementation for the transcription model and training code.
https://github.com/KinWaiCheuk/Jointist


## Repository Layout

- config/: Hydra configs for extraction, VAE training, and inference.
- scripts/extract/extract.sh: Example extraction launcher.
- scripts/vae/train.sh: Example VAE training launcher.
- scripts/inference/run.sh: Example inference launcher.
- scripts/setup/docker.sh: Docker-based environment bootstrap.
- src/extract/: DDSP/NSynth feature extraction code.
- src/vae/: VAE model, dataset loader, trainer.
- src/inference/: Clustering, tau estimation, MIDI parsing, synthesis, packaging.

## System Requirements

Recommended (validated path):

- Linux
- NVIDIA GPU (for extraction and VAE training)
- CUDA-compatible drivers
- Python 3.8.10 (recommended for TensorFlow 2.10 and DDSP 3.7 compatibility)

Important package constraints in this repo:

- tensorflow==2.10.0
- ddsp==3.7.0
- torch==2.0.1 in pyproject.toml

If you use different versions, expect behavior drift.

## Environment Setup
Run from repository root:

```bash
bash scripts/setup/docker.sh
```
This script builds the development container, creates a virtual environment, and installs dependencies.

## Data Prerequisites

You need the following data sources:

1. NSynth TFDS source for extraction stage.
2. Extracted feature directories for VAE:
	- train directory containing files like feature_*.npy and label_*.npy
	- valid directory containing files like feature_*.npy and label_*.npy
	- test directory containing files like feature_*.npy and label_*.npy
3. Lakh MIDI dataset root directory for inference.
4. split_info.pkl used by inference to determine train/test/validation split.
5. Optional dismiss list file for bad MIDI paths.

## Stage 1: Feature Extraction (NSynth -> DDSP/Harmonic)

Use the provided example script:

```bash
bash scripts/extract/extract.sh
```

By default this script demonstrates a short run with dataset.max_samples=10.
For full extraction, increase or remove this override.

You usually run this stage per split, for example by changing:

- extraction.split=train
- extraction.split=valid
- extraction.split=test

in the command overrides.

Output files are created under paths.output_root/split and include:

- feature_*.npy
- label_*.npy
- f0_*.npy
- loudness_*.npy
- amps_*.npy
- noise_*.npy
- f0_confidence_*.npy
- metadata_*.json

## Stage 2: VAE Training

Run:

```bash
bash scripts/vae/train.sh
```

Default training config entrypoint:

- config name: vae
- config file: config/vae.yaml

Main outputs:

- trained checkpoints under outputs/
- Hydra run config snapshot under .hydra/

Keep both of these for the inference stage:

1. best model checkpoint path
2. corresponding .hydra/config.yaml path

## Stage 3: Inference and Synthetic Data Packaging

Run:

```bash
bash scripts/inference/run.sh
```

The current script is designed for local testing via Hydra overrides.
It overrides private paths (checkpoint, split_info, dataset roots, feature dirs).
This is acceptable for internal experiments.

For public release, keep config/inference.yaml generic and remove machine-specific absolute paths from scripts.

### What the inference stage does

1. Loads VAE checkpoint and training config.
2. Loads latent embeddings from the designated feature split(s) for clustering
3. Builds or loads clustering cache.
4. Computes recommended tau from cluster spread.
5. Parses split_info and MIDI files.
6. Samples latent vectors and decodes harmonic distributions.
7. Synthesizes per-track audio and cuts fixed-length segments.
8. Exports packaged files and metadata.

### Inference outputs

Under output.root_dir:

- packed_waveforms/train|test|validation/sample*/waveform.flac
- packed_pkl/train|test|validation/sample*.pkl
- metadata.json
- summary.json
- cluster_cache/*

These outputs are the intended handoff point before AMT model ingestion.

## Typical End-to-End Order

1. Extract NSynth features (train/valid/test).
2. Train VAE.
3. Run inference pseudo-data generation.
4. Feed generated packed data into your AMT model pipeline (not covered here).

## Configuration Notes

- config/inference.yaml is intentionally placeholder-based.
- scripts/inference/run.sh is the practical place to override local paths during experimentation.
- If feature directory paths in training config differ from inference environment, always override with:
  - data.train_data_dir
  - data.valid_data_dir
  - data.test_data_dir

## Current Scope and Next Step

This repository currently documents and implements preprocessing, representation learning, and pseudo-audio generation.
The downstream AMT training/finetuning stage that consumes packed_waveforms and packed_pkl should be implemented in your target AMT repository.
