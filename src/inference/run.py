import json
import logging
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.inference.clustering import HarmonicClusterAnalyzer, load_or_create_clustering_data
from src.inference.io import _exec_one, _init_worker
from src.inference.midi import load_dismiss_midis, parse_midi_notes_mido, parse_split_info
from src.inference.tau import recommend_tau
from src.utils.logging import setup_inference_logging
from src.utils.reproducibility import make_deterministic, set_seed

logger = logging.getLogger(__name__)

ALLOWED_SPLITS = ("train", "test", "validation")


def gaussian_sample_localcov(center, members, n, tau, rng):
    """Sample n points from N(center, (tau^2) * Sigma_cluster)."""
    x_centered = members - members.mean(axis=0, keepdims=True)
    sigma = np.cov(x_centered, rowvar=False, ddof=1)
    eps = 1e-6 * np.trace(sigma) / sigma.shape[0]
    sigma = sigma + eps * np.eye(sigma.shape[0])

    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.clip(eigvals, 1e-12, None)
        chol = eigvecs @ np.diag(np.sqrt(eigvals))

    z = rng.standard_normal(size=(n, sigma.shape[0]))
    return center + tau * (z @ chol.T)


def _resolve_tau(cfg: DictConfig, members_by_cluster, centers):
    tau_cfg = cfg.tau
    tau_stats = recommend_tau(
        members_by_cluster=members_by_cluster,
        centers=centers,
        clamp_min=tau_cfg.get("clamp_min", None),
        clamp_max=tau_cfg.get("clamp_max", None),
        round_digits=tau_cfg.get("round_digits", None),
    )

    mode = str(tau_cfg.get("mode", "auto"))
    if mode == "manual":
        tau_used = float(tau_cfg.manual_value)
    else:
        tau_used = float(tau_stats["recommended_tau"])

    tau_stats["mode"] = mode
    tau_stats["tau_used"] = tau_used
    return tau_used, tau_stats


def _validate_split_name(split_name: str) -> str:
    if split_name not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of {ALLOWED_SPLITS}.")
    return split_name


def _assign_program_cluster_params(programs, rng, k: int):
    available_clusters = list(range(k))
    rng.shuffle(available_clusters)

    program_cluster_params = {}
    for i, program in enumerate(sorted(programs)):
        cluster_id = available_clusters[i % len(available_clusters)]
        magnitudes_level = rng.uniform(0.0, 0.01)
        ir_level = rng.uniform(0.0, 0.5)
        program_cluster_params[program] = (cluster_id, magnitudes_level, ir_level)

    return program_cluster_params


def _decode_program_harmonic_distributions(
    program_cluster_params,
    members_by_cluster,
    centers,
    analyzer: HarmonicClusterAnalyzer,
    tau_used: float,
    rng,
):
    program_harmonic_dists = {}
    for program, (cluster_id, _, _) in program_cluster_params.items():
        members = members_by_cluster[cluster_id]
        if members.shape[0] < 2:
            continue

        center = centers[cluster_id]
        sample = gaussian_sample_localcov(center, members, 1, tau_used, rng)[0]

        with torch.no_grad():
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(analyzer.device)
            harmonic_dist = analyzer.model.decode(sample_tensor).cpu().numpy()[0]
            harmonic_dist = np.clip(harmonic_dist, 0.0, None)
            harmonic_dist = harmonic_dist / (harmonic_dist.sum() + 1e-12)

        program_harmonic_dists[program] = harmonic_dist

    return program_harmonic_dists


def _build_task(
    rel_path: str,
    split_name: str,
    notes,
    program_harmonic_dists,
    program_cluster_params,
    out_dir: Path,
    base_sample_idx: int,
    sample_rate: int,
    segment_duration: float,
):
    return (
        rel_path,
        split_name,
        notes,
        program_harmonic_dists,
        program_cluster_params,
        str(out_dir),
        base_sample_idx,
        sample_rate,
        segment_duration,
    )


@hydra.main(version_base=None, config_path="../../config", config_name="inference")
def main(cfg: DictConfig) -> None:
    setup_inference_logging(logging.INFO)

    set_seed(int(cfg.runtime.seed))
    if bool(cfg.runtime.get("deterministic", False)):
        make_deterministic()

    if bool(cfg.runtime.get("disable_tf_gpu", True)):
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

    out_dir = Path(cfg.output.root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test", "validation"]:
        (out_dir / "packed_waveforms" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "packed_pkl" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.clustering.random_state))

    analyzer = HarmonicClusterAnalyzer(
        model_path=str(cfg.model.checkpoint_path),
        config_path=str(cfg.model.train_config_path),
        output_dir=str(out_dir),
        normalizer_path=str(cfg.data.get("normalizer_path", "")) or None,
    )

    _, _, members_by_cluster, centers = load_or_create_clustering_data(analyzer, str(out_dir), cfg)
    logger.info("Clustering completed: %d clusters", int(cfg.clustering.k))

    tau_used, tau_stats = _resolve_tau(cfg, members_by_cluster, centers)
    logger.info("Recommended tau(raw): %.6f", tau_stats["recommended_tau_raw"])
    logger.info("Tau used: %.6f", tau_used)

    with open(str(cfg.data.split_info_pkl), "rb") as f:
        split_info_raw = pickle.load(f)
    split_info = parse_split_info(split_info_raw)

    dismiss_midis = load_dismiss_midis(cfg.data.get("dismiss_midis", None))
    if dismiss_midis:
        logger.info("Loaded %d files to dismiss", len(dismiss_midis))

    sample_count = {split: 0 for split in ALLOWED_SPLITS}
    tasks = []
    processed_count = 0

    max_midis = int(cfg.synthesis.max_midis)
    sample_rate = int(cfg.synthesis.sample_rate)
    segment_duration = float(cfg.synthesis.segment_duration)

    for midi_info in split_info:
        if processed_count >= max_midis:
            break

        rel_path = midi_info.get("path")
        split_name = _validate_split_name(str(midi_info.get("split", "train")))
        if not rel_path:
            continue

        midi_path = os.path.join(str(cfg.data.lakh_dataset_dir), rel_path)
        if rel_path in dismiss_midis or not os.path.exists(midi_path):
            continue

        notes = parse_midi_notes_mido(midi_path)
        if not notes:
            continue

        programs = set(note["program"] for note in notes)
        if not programs:
            continue

        program_cluster_params = _assign_program_cluster_params(programs, rng, int(cfg.clustering.k))
        program_harmonic_dists = _decode_program_harmonic_distributions(
            program_cluster_params,
            members_by_cluster,
            centers,
            analyzer,
            tau_used,
            rng,
        )

        if not program_harmonic_dists:
            continue

        base_sample_idx = sample_count[split_name]
        tasks.append(
            _build_task(
                rel_path,
                split_name,
                notes,
                program_harmonic_dists,
                program_cluster_params,
                out_dir,
                base_sample_idx,
                sample_rate,
                segment_duration,
            )
        )

        max_offset = max(note["offset"] for note in notes)
        sample_count[split_name] += max(1, int(max_offset // segment_duration))

        processed_count += 1

    logger.info("Starting parallel synthesis for %d MIDI files", len(tasks))
    metadata = []

    if tasks:
        max_workers = min(int(cfg.synthesis.num_workers), len(tasks))
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            mp_context=spawn_context,
        ) as executor:
            future_to_task = {executor.submit(_exec_one, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing MIDIs"):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        metadata.extend(result)
                except Exception as exc:
                    logger.error("Error processing %s: %s", task[0], exc)

    if bool(cfg.output.save_metadata):
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if bool(cfg.output.save_summary):
        summary = {
            "k": int(cfg.clustering.k),
            "tau": tau_stats,
            "num_tasks": len(tasks),
            "num_samples": len(metadata),
            "seed": int(cfg.runtime.seed),
            "model_checkpoint": str(cfg.model.checkpoint_path),
            "train_config": str(cfg.model.train_config_path),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    logger.info("Processing complete: %d generated samples", len(metadata))


if __name__ == "__main__":
    main()
