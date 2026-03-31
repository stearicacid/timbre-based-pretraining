import os
import tensorflow as tf
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
import logging

from utils.logging import setup_logging
from extract.dataset import NSynthTfds
from extract.workers import process_single_sample
from extract.save import save_results

logger = logging.getLogger(__name__)
setup_logging()

def _select(cfg: DictConfig, key: str, default=None):
    """Safely fetch nested config values with a default."""
    value = OmegaConf.select(cfg, key)
    return default if value is None else value

def extract_nsynth_features(cfg, split="train", feature_type="harmonic", max_samples=None, workers_per_gpu=None):
    """
    extract harmonic distribution and metadata from NSynth dataset using GPU parallel processing
    """
    
    output_root = _select(cfg, "paths.output_root", getattr(cfg, "output_dir", "outputs/extract"))
    output_dir = os.path.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting EXTENDED GPU-PARALLEL NSynth processing ({split} split)")
    
    parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_gpu:
        logger.info(f"Using parent GPU setting: {parent_gpu}")
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parent_gpu.split(',')]
        num_physical_gpus = len(gpu_ids)
    else:
        physical_gpus = tf.config.list_physical_devices('GPU')
        num_physical_gpus = len(physical_gpus)
        gpu_ids = list(range(num_physical_gpus))
    
    if num_physical_gpus == 0:
        logger.error("No GPUs found.")
        return {}

    logger.info(f"Using {num_physical_gpus} GPU(s): {gpu_ids}")

    if workers_per_gpu is None:
        workers_per_gpu = _select(cfg, "runtime.workers_per_gpu", getattr(cfg, "workers_per_gpu", 2))
    
    total_workers = num_physical_gpus * workers_per_gpu
    logger.info(f"Starting {workers_per_gpu} workers per GPU, for a total of {total_workers} workers.")

    # get dataset
    data_provider = NSynthTfds(
        name=cfg.dataset.name,
        split=split,
        data_dir=_select(cfg, "dataset.data_dir", None),
        sample_rate=_select(cfg, "audio.sample_rate", 16000),
        frame_rate=_select(cfg, "dataset.frame_rate", 250),
        include_note_labels=_select(cfg, "dataset.include_note_labels", True),
    )

    try:
        batch_size = max(1, num_physical_gpus)
        dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False)
        logger.info("Successfully loaded NSynth dataset")
    except Exception:
        logger.exception(f"Failed to load NSynth dataset")
        raise
    
    family_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 
                    'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

    if max_samples is None:
        max_samples = _select(cfg, "dataset.max_samples", 1000000)
    if max_samples is None:
        max_samples = 1000000
    
    existing_files = {f[8:-4] for f in os.listdir(output_dir) if f.startswith('feature_') and f.endswith('.npy')}
    logger.info(f"Found {len(existing_files)} existing processed files.")
    
    # tasks to process
    tasks_to_process = []
    sample_count = 0
    
    logger.info("Preparing task list...")
    for batch in dataset.take((max_samples + batch_size - 1) // batch_size):
        if sample_count >= max_samples:
            break

        audio_tf = batch['audio']
        
        for j in range(audio_tf.shape[0]):
            if sample_count >= max_samples:
                break
            
            current_file_id = f"tfds_{split}_{sample_count}"
            
            if current_file_id in existing_files:
                sample_count += 1
                continue

            audio_np = audio_tf[j].numpy()
            
            family_id = batch['instrument_family'][j].numpy() if 'instrument_family' in batch else -1
            instrument_family = family_names[family_id] if 0 <= family_id < len(family_names) else f"family_{family_id}"
            
            source_index = batch['instrument_source'][j].numpy() if 'instrument_source' in batch else -1
            source_names = ['acoustic', 'electronic', 'synthetic']
            instrument_source = source_names[source_index] if 0 <= source_index < len(source_names) else f"source_{source_index}"

            pitch = batch['pitch'][j].numpy() if 'pitch' in batch else -1
            
            params = {
                'audio_data': audio_np,
                'file_id': current_file_id,
                'instrument_family': instrument_family,
                'instrument_name': f"instrument_{family_id}",
                'label': family_id,
                'instrument_source': instrument_source,
                'instrument_source_idx': int(source_index),
                'pitch': int(pitch),
                'cfg': cfg,
                'feature_type': feature_type,
                'output_dir': output_dir,
                'gpu_id': gpu_ids[sample_count % num_physical_gpus] if not parent_gpu else None
            }
            
            tasks_to_process.append(params)
            sample_count += 1

    logger.info(f"Prepared {len(tasks_to_process)} tasks for processing")
    
    all_paths = {
        'feature_paths': [], 'f0_paths': [], 'loudness_paths': [],
        'amps_paths': [], 'noise_paths': [], 'f0_confidence_paths': [], 'label_paths': []
    }
    
    completed_tasks = 0
    failed_tasks = 0

    save_batch_size = _select(cfg, "runtime.save_batch_size", None)
    SAVE_BATCH_SIZE = save_batch_size or (workers_per_gpu * num_physical_gpus)
    results_batch = []

    try:
        with mp.Pool(
            processes=total_workers,
            maxtasksperchild=_select(cfg, "runtime.multiprocessing.pool.maxtasksperchild", 1),
        ) as pool:
            logger.info("Starting parallel processing with ProcessPool...")

            with tqdm(total=len(tasks_to_process), desc="Processing samples") as pbar:
                for result in pool.imap_unordered(
                    process_single_sample,
                    tasks_to_process,
                    chunksize=_select(cfg, "runtime.multiprocessing.pool.chunksize", 1),
                ):
                    if result and result["status"] == "success":
                        results_batch.append(result) 
                        completed_tasks += 1
                    else:
                        logger.warning(f"Task failed: {result.get('file_id', 'unknown')}: {result.get('error', 'unknown error')}")
                        failed_tasks += 1
                    pbar.update(1)

                    if len(results_batch) >= SAVE_BATCH_SIZE:
                        logger.info(f"Saving batch of {len(results_batch)} files...")
                        save_results(results_batch, output_dir, all_paths)
                        results_batch.clear()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
        raise
    except Exception:
        logger.exception(f"Unexpected error during parallel processing")
        raise
    finally:
        if results_batch:
            logger.info(f"Saving final batch of {len(results_batch)} files...")
            save_results(results_batch, output_dir, all_paths)
            results_batch.clear()

    logger.info(f"EXTENDED processing completed: {completed_tasks} successful, {failed_tasks} failed")

    for file_id in existing_files:
        all_paths['feature_paths'].append(os.path.join(output_dir, f"feature_{file_id}.npy"))
        all_paths['f0_paths'].append(os.path.join(output_dir, f"f0_{file_id}.npy"))
        all_paths['loudness_paths'].append(os.path.join(output_dir, f"loudness_{file_id}.npy"))
        all_paths['amps_paths'].append(os.path.join(output_dir, f"amps_{file_id}.npy"))
        all_paths['noise_paths'].append(os.path.join(output_dir, f"noise_{file_id}.npy"))
        all_paths['f0_confidence_paths'].append(os.path.join(output_dir, f"f0_confidence_{file_id}.npy"))
        all_paths['label_paths'].append(os.path.join(output_dir, f"label_{file_id}.npy"))
        
    return all_paths


@hydra.main(config_path="../../config", config_name="extract", version_base=None)
def main(cfg: DictConfig):
    start_method = _select(cfg, "runtime.multiprocessing.start_method", "spawn")
    mp.set_start_method(start_method, force=True)
    
    should_extract = _select(cfg, "extraction.save_features", getattr(cfg, "save_extended_features", False))
    if should_extract:
        logger.info("Using extended feature extraction with memory management")
        split = _select(cfg, "extraction.split", getattr(cfg, "split", "train"))
        feature_type = _select(cfg, "feature.type", getattr(cfg, "feature_type", "harmonic"))
        extract_nsynth_features(
            cfg,
            split,
            feature_type,
            _select(cfg, "dataset.max_samples", getattr(cfg, "max_samples", None)),
            _select(cfg, "runtime.workers_per_gpu", getattr(cfg, "workers_per_gpu", None)),
        )

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()