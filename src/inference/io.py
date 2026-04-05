import logging
import os
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # Keep TF workers CPU-only to avoid per-process GPU context allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass


def _exec_one(task):
    """Execute one prepared synthesis task and return metadata rows."""
    (
        midi_rel_path,
        split_name,
        notes,
        program_harmonic_dists,
        program_cluster_params,
        out_dir,
        base_sample_idx,
        sample_rate,
        segment_duration,
    ) = task

    from src.inference.synthesis import cut_into_segments, synthesize_midi_track

    audio = synthesize_midi_track(
        notes,
        program_harmonic_dists,
        program_cluster_params,
        sample_rate=sample_rate,
    )
    if len(audio) == 0:
        return []

    segments = cut_into_segments(
        audio,
        notes,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
    )
    if not segments:
        return []

    meta_chunk = []
    for seg_idx, segment in enumerate(segments):
        sample_name = f"sample{base_sample_idx + seg_idx + 1}"
        if len(segment["audio"]) == 0 or np.all(segment["audio"] == 0):
            continue

        audio_path = Path(out_dir) / "packed_waveforms" / split_name / sample_name / "waveform.flac"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), segment["audio"], sample_rate)

        pkl_path = Path(out_dir) / "packed_pkl" / split_name / f"{sample_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "midi_notes": segment["midi"],
                    "program_cluster_params": program_cluster_params,
                    "harmonic_distributions": program_harmonic_dists,
                },
                f,
            )

        meta_chunk.append(
            {
                "sample_name": sample_name,
                "midi_file": midi_rel_path,
                "start_time": float(segment["start_time"]),
                "end_time": float(segment["end_time"]),
                "split": split_name,
                "cluster_assignments": {
                    str(program): int(values[0]) for program, values in program_cluster_params.items()
                },
                "programs": [int(program) for program in program_cluster_params.keys()],
            }
        )

    return meta_chunk
