import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import multiprocessing as mp
import json
from datetime import datetime
import gc
import psutil

import ddsp
from ddsp.training import models, preprocessing, decoders
from ddsp import spectral_ops
from ddsp.training import train_util, data
import tensorflow_datasets as tfds
from extract.save import save_results
from utils.utils import get_memory_usage, cleanup_tensorflow_memory
from extract.dataset import CustomNSynthTfds
from extract.workers import process_single_sample

# Legacy note (translated to English).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_preparation_gpu_parallel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# def get_memory_usage():
# Legacy note (translated to English).
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
# Legacy note (translated to English).

# def cleanup_tensorflow_memory():
# Legacy note (translated to English).
#     try:
# Legacy note (translated to English).
#         tf.keras.backend.clear_session()
        
# Legacy note (translated to English).
#         if tf.config.list_physical_devices('GPU'):
#             try:
#                 tf.config.experimental.reset_memory_stats('GPU:0')
#             except Exception as e:
#                 logger.debug(f"GPU memory reset warning: {e}")
        
# Legacy note (translated to English).
#         gc.collect()
        
# Legacy note (translated to English).
#         spectral_ops.reset_crepe()
        
#     except Exception as e:
#         logger.warning(f"TensorFlow memory cleanup warning: {e}")

# class CustomNSynthTfds(data.TfdsProvider):
# Legacy note (translated to English).
    
#     def __init__(self,
#                  name='nsynth/full:2.3.3',
#                  split='train',
#                  data_dir='/mlnas/rin/tensorflow_datasets',
#                  sample_rate=16000,
#                  frame_rate=250,
#                  include_note_labels=True):
#         """
# Legacy note (translated to English).
        
#         Args:
#             name: TFDS dataset name
#             split: Dataset split
# Legacy note (translated to English).
# Legacy note (translated to English).
# Legacy note (translated to English).
# Legacy note (translated to English).
#         """
#         self._include_note_labels = include_note_labels
#         super().__init__(name, split, data_dir, sample_rate, frame_rate)
        
# Legacy note (translated to English).
#         self.family_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 
#                             'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    
#     def get_dataset(self, shuffle=True):
# Legacy note (translated to English).
#         def preprocess_ex(ex):
# Legacy note (translated to English).
# Legacy note (translated to English).
#             audio = ex['audio']
            
# Legacy note (translated to English).
#             audio_16k = tf.cast(audio, tf.float32)
            
# Legacy note (translated to English).
# Legacy note (translated to English).
#             audio_length = tf.shape(audio_16k)[0]
            
# Legacy note (translated to English).
#             audio_length_float = tf.cast(audio_length, tf.float32)
#             sample_rate_float = tf.cast(self.sample_rate, tf.float32)
#             frame_rate_float = tf.cast(self.frame_rate, tf.float32)
            
#             expected_frames = tf.cast(tf.math.ceil(audio_length_float / (sample_rate_float / frame_rate_float)), tf.int32)
            
# Legacy note (translated to English).
#             f0_hz = tf.zeros([expected_frames], dtype=tf.float32)
#             f0_confidence = tf.zeros([expected_frames], dtype=tf.float32)
#             loudness_db = tf.zeros([expected_frames], dtype=tf.float32)
            
# Legacy note (translated to English).
#             ex_out = {
#                 'audio': audio_16k,
#                 'f0_hz': f0_hz,
#                 'f0_confidence': f0_confidence,
#                 'loudness_db': loudness_db,
#             }
            
#             if self._include_note_labels:
#                 ex_out.update({
#                     'pitch': ex['pitch'],
#                     'instrument_source': ex['instrument']['source'],
#                     'instrument_family': ex['instrument']['family'],
#                     'instrument': ex['instrument']['label'],
#                 })
            
#             return ex_out
        
#         dataset = super().get_dataset(shuffle)
#         dataset = dataset.map(preprocess_ex, num_parallel_calls=tf.data.AUTOTUNE)
#         return dataset

# def extract_harmonic_distribution_with_controls(audio_np, cfg):
#     """
# Legacy note (translated to English).
#     """
#     func_start_time = time.time()
# Legacy note (translated to English).
    
#     sample_rate = getattr(cfg.audio, 'sample_rate', 16000)
#     max_length = getattr(cfg.audio, 'max_length', 64000)
    
#     if len(audio_np) > max_length:
#         audio_np = audio_np[:max_length]
#     elif len(audio_np) < max_length:
#         audio_np = np.pad(audio_np, (0, max_length - len(audio_np)), 'constant')

#     audio_tf = tf.convert_to_tensor([audio_np], dtype=tf.float32)
    
# Legacy note (translated to English).
#     try:
#         f0_start_time = time.time()
#         spectral_ops.reset_crepe()
#         f0_hz_np, f0_confidence_np = spectral_ops.compute_f0(audio_tf[0], frame_rate=250, viterbi=True)
#         f0_hz_tf = tf.expand_dims(f0_hz_np, 0)
#         loudness_db_tf = spectral_ops.compute_loudness(audio_tf, sample_rate)
# Legacy note (translated to English).
#     except Exception as e:
#         logger.warning(f"Feature extraction error: {e}")
#         time_steps = 1000
#         f0_hz_tf = tf.ones([1, time_steps], dtype=tf.float32) * 440.0
#         loudness_db_tf = tf.ones([1, time_steps], dtype=tf.float32) * -30.0
#         f0_confidence_np = tf.ones([time_steps], dtype=tf.float32) * 0.5
    
#     cleanup_start_time = time.time()
#     cleanup_tensorflow_memory()
# Legacy note (translated to English).
    
#     try:
#         strategy_start_time = time.time()
# Legacy note (translated to English).
#         available_gpus = tf.config.list_physical_devices('GPU')
#         parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
#         if len(available_gpus) > 1 or (parent_gpu and ',' in parent_gpu):
# Legacy note (translated to English).
#             with tf.device('/GPU:0'):
#                 strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
#         else:
#             strategy = train_util.get_strategy()
# Legacy note (translated to English).
        
#         dataset_start_time = time.time()
#         dataset = tf.data.Dataset.from_tensors({
#             'audio': audio_tf, 'f0_hz': f0_hz_tf, 'loudness_db': loudness_db_tf
#         }).repeat()

#         TIME_STEPS = cfg.model.time_steps
#         n_samples = audio_tf.shape[1]
# Legacy note (translated to English).
        
#         model_build_start_time = time.time()
#         with strategy.scope():
#             preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=TIME_STEPS)
#             decoder = decoders.RnnFcDecoder(
#                 rnn_channels=cfg.model.decoder.rnn_channels, rnn_type=cfg.model.decoder.rnn_type,
#                 ch=cfg.model.decoder.ch, layers_per_stack=cfg.model.decoder.layers_per_stack,
#                 input_keys=('ld_scaled', 'f0_scaled'),
#                 output_splits=(('amps', 1), ('harmonic_distribution', cfg.model.harmonic.n_harmonics),
#                                ('noise_magnitudes', cfg.model.harmonic.n_harmonics))
#             )
            
#             harmonic = ddsp.synths.Harmonic(
#                 n_samples=n_samples, 
#                 sample_rate=sample_rate, 
#                 scale_fn=ddsp.core.exp_sigmoid,
#                 name='harmonic'
#             )
            
#             noise = ddsp.synths.FilteredNoise(
#                 n_samples=n_samples, 
#                 window_size=getattr(cfg.model.noise, 'window_size', 0),
#                 initial_bias=getattr(cfg.model.noise, 'initial_bias', -10.0),
#                 name='noise'
#             )
            
#             add = ddsp.processors.Add(name='add')

#             dag = [
#                 (harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
#                 (noise, ['noise_magnitudes']),
#                 (add, ['noise/signal', 'harmonic/signal'])
#             ]
            
#             processor_group = ddsp.processors.ProcessorGroup(dag=dag, name='processor_group')
#             spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0)
#             model = models.Autoencoder(preprocessor=preprocessor, encoder=None, decoder=decoder, processor_group=processor_group, losses=[spectral_loss])
#             trainer = ddsp.training.trainers.Trainer(model, strategy, learning_rate=cfg.model.training.learning_rate)
# Legacy note (translated to English).

#         dataset_prep_start_time = time.time()
#         dataset = trainer.distribute_dataset(dataset)
#         dataset_iter = iter(dataset)
# Legacy note (translated to English).
        
#         training_start_time = time.time()
#         n_steps = cfg.model.training.n_steps
# Legacy note (translated to English).
#         for step in range(n_steps):
#             step_start = time.time()
#             trainer.train_step(dataset_iter)
# Legacy note (translated to English).
# Legacy note (translated to English).
# Legacy note (translated to English).
        
#         inference_start_time = time.time()
#         batch = next(dataset_iter)
#         controls = model(batch)
        
#         harmonic_distribution_np = controls['harmonic_distribution'][0].numpy()
#         amps_np = controls['amps'][0].numpy()
#         noise_magnitudes_np = controls['noise_magnitudes'][0].numpy()
# Legacy note (translated to English).
        
# Legacy note (translated to English).
# Legacy note (translated to English).
#         if isinstance(f0_hz_np, tf.Tensor):
#             f0_hz_original = f0_hz_np.numpy()
#         else:
#             f0_hz_original = f0_hz_np
            
#         loudness_db_original = loudness_db_tf[0].numpy()
        
# Legacy note (translated to English).
#         if isinstance(f0_confidence_np, tf.Tensor):
#             f0_confidence_original = f0_confidence_np.numpy()
#         else:
#             f0_confidence_original = f0_confidence_np
        
#         total_time = time.time() - func_start_time
# Legacy note (translated to English).
        
#         return {
#             'harmonic_distribution': harmonic_distribution_np,
#             'amps': amps_np,
#             'noise_magnitudes': noise_magnitudes_np,
#             'f0_hz': f0_hz_original,
#             'loudness_db': loudness_db_original,
#             'f0_confidence': f0_confidence_original
#         }
        
#     finally:
# Legacy note (translated to English).
# Legacy note (translated to English).

def extract_nsynth_data_gpu_parallel_extended(cfg, split="train", feature_type="harmonic", max_samples=None, workers_per_gpu=None):
    """
    Extended GPU-parallel processing with memory management.
    """
    print("[DEBUG] extract_nsynth_data_gpu_parallel_extended started")
    
    output_dir = os.path.join(cfg.output_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    
    start_time_total = time.time()
    logger.info(f"Starting EXTENDED GPU-PARALLEL NSynth processing ({split} split)")
    
    # Legacy note (translated to English).
    parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_gpu:
        logger.info(f"Using parent GPU setting: {parent_gpu}")
        # Legacy note (translated to English).
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parent_gpu.split(',')]
        num_physical_gpus = len(gpu_ids)
    else:
        # Legacy note (translated to English).
        physical_gpus = tf.config.list_physical_devices('GPU')
        num_physical_gpus = len(physical_gpus)
        gpu_ids = list(range(num_physical_gpus))
    
    if num_physical_gpus == 0:
        logger.error("No GPUs found.")
        return {}

    logger.info(f"Using {num_physical_gpus} GPU(s): {gpu_ids}")

    if workers_per_gpu is None:
        workers_per_gpu = getattr(cfg, 'workers_per_gpu', 2)
    
    total_workers = num_physical_gpus * workers_per_gpu
    logger.info(f"Starting {workers_per_gpu} workers per GPU, for a total of {total_workers} workers.")

    # Legacy note (translated to English).
    data_provider = CustomNSynthTfds(
        name=cfg.dataset.name,
        split=split,
        data_dir=cfg.dataset.data_dir,
        sample_rate=cfg.audio.sample_rate,
        frame_rate=250
    )

    try:
        batch_size = max(1, num_physical_gpus)
        dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False)
        logger.info("Successfully loaded NSynth dataset")
    except Exception as e:
        logger.error(f"Failed to load NSynth dataset: {e}")
        raise
    
    family_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 
                    'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

    if max_samples is None:
        max_samples = getattr(cfg.dataset, 'max_samples', 1000000)
    
    existing_files = {f[8:-4] for f in os.listdir(output_dir) if f.startswith('feature_') and f.endswith('.npy')}
    logger.info(f"Found {len(existing_files)} existing processed files.")
    
    # Legacy note (translated to English).
    tasks_to_process = []
    sample_count = 0
    
    logger.info("Preparing task list...")
    task_prep_start_time = time.time()
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
                # Legacy note (translated to English).
                'gpu_id': gpu_ids[sample_count % num_physical_gpus] if not parent_gpu else None
            }
            
            tasks_to_process.append(params)
            sample_count += 1
    
    print(f"[DEBUG] Task preparation time: {time.time() - task_prep_start_time:.3f}s")
    logger.info(f"Prepared {len(tasks_to_process)} tasks for processing")
    
    all_paths = {
        'feature_paths': [], 'f0_paths': [], 'loudness_paths': [],
        'amps_paths': [], 'noise_paths': [], 'f0_confidence_paths': [], 'label_paths': []
    }
    
    completed_tasks = 0
    failed_tasks = 0
    
    # Legacy note (translated to English).
    SAVE_BATCH_SIZE = workers_per_gpu * num_physical_gpus # Save once per total worker count.
    results_batch = []

    try:
        pool_start_time = time.time()
        with mp.Pool(processes=total_workers, maxtasksperchild=1) as pool:
            logger.info("Starting parallel processing with ProcessPool...")
            print(f"[DEBUG] Pool startup time: {time.time() - pool_start_time:.3f}s")

            with tqdm(total=len(tasks_to_process), desc="Processing samples") as pbar:
                for result in pool.imap_unordered(process_single_sample, tasks_to_process, chunksize=1):
                    if result and result["status"] == "success":
                        results_batch.append(result) # Add result to the current save batch.
                        completed_tasks += 1
                    else:
                        logger.error(f"Task failed: {result.get('file_id', 'unknown')}: {result.get('error', 'unknown error')}")
                        failed_tasks += 1
                    pbar.update(1)

                    # Legacy note (translated to English).
                    if len(results_batch) >= SAVE_BATCH_SIZE:
                        logger.info(f"Saving batch of {len(results_batch)} files...")
                        save_results(results_batch, output_dir, all_paths)
                        results_batch.clear()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
    except Exception as e:
        import traceback
        logger.error(f"Error in processing: {e}\n{traceback.format_exc()}")
    finally:
        # Legacy note (translated to English).
        if results_batch:
            logger.info(f"Saving final batch of {len(results_batch)} files...")
            save_results(results_batch, output_dir, all_paths)
            results_batch.clear()

    processing_time = time.time() - start_time_total
    logger.info(f"EXTENDED processing completed: {completed_tasks} successful, {failed_tasks} failed")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    # Legacy note (translated to English).
    for file_id in existing_files:
        all_paths['feature_paths'].append(os.path.join(output_dir, f"feature_{file_id}.npy"))
        all_paths['f0_paths'].append(os.path.join(output_dir, f"f0_{file_id}.npy"))
        all_paths['loudness_paths'].append(os.path.join(output_dir, f"loudness_{file_id}.npy"))
        all_paths['amps_paths'].append(os.path.join(output_dir, f"amps_{file_id}.npy"))
        all_paths['noise_paths'].append(os.path.join(output_dir, f"noise_{file_id}.npy"))
        all_paths['f0_confidence_paths'].append(os.path.join(output_dir, f"f0_confidence_{file_id}.npy"))
        all_paths['label_paths'].append(os.path.join(output_dir, f"label_{file_id}.npy"))
        
    return all_paths

# def save_results_batch(results_list, output_dir, all_paths):
#     """
# Legacy note (translated to English).
#     """
#     save_start_time = time.time()
# Legacy note (translated to English).
    
#     for i, result in enumerate(results_list):
#         file_save_start = time.time()
#         file_id = result['file_id']
#         features_dict = result['data']
#         metadata = result['metadata']
        
# Legacy note (translated to English).
#         path_gen_start = time.time()
#         feature_path = os.path.join(output_dir, f"feature_{file_id}.npy")
#         f0_path = os.path.join(output_dir, f"f0_{file_id}.npy")
#         loudness_path = os.path.join(output_dir, f"loudness_{file_id}.npy")
#         amps_path = os.path.join(output_dir, f"amps_{file_id}.npy")
#         noise_path = os.path.join(output_dir, f"noise_{file_id}.npy")
#         f0_confidence_path = os.path.join(output_dir, f"f0_confidence_{file_id}.npy")
#         label_path = os.path.join(output_dir, f"label_{file_id}.npy")
#         metadata_path = os.path.join(output_dir, f"metadata_{file_id}.json")
# Legacy note (translated to English).

# Legacy note (translated to English).
#         numpy_save_start = time.time()
#         np.save(feature_path, features_dict['harmonic_distribution'])
#         np.save(f0_path, features_dict['f0_hz'])
#         np.save(loudness_path, features_dict['loudness_db'])
#         np.save(amps_path, features_dict['amps'])
#         np.save(noise_path, features_dict['noise_magnitudes'])
#         np.save(f0_confidence_path, features_dict['f0_confidence'])
#         np.save(label_path, np.array(metadata['label']))
#         numpy_save_time = time.time() - numpy_save_start
# Legacy note (translated to English).
        
# Legacy note (translated to English).
#         json_save_start = time.time()
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
#         json_save_time = time.time() - json_save_start
# Legacy note (translated to English).

# Legacy note (translated to English).
#         path_record_start = time.time()
#         all_paths['feature_paths'].append(feature_path)
#         all_paths['f0_paths'].append(f0_path)
#         all_paths['loudness_paths'].append(loudness_path)
#         all_paths['amps_paths'].append(amps_path)
#         all_paths['noise_paths'].append(noise_path)
#         all_paths['f0_confidence_paths'].append(f0_confidence_path)
#         all_paths['label_paths'].append(label_path)
#         path_record_time = time.time() - path_record_start
# Legacy note (translated to English).
        
#         file_total_time = time.time() - file_save_start
# Legacy note (translated to English).
        
# Legacy note (translated to English).
#         if (i + 1) % 10 == 0 or (i + 1) == len(results_list):
# Legacy note (translated to English).
    
#     total_save_time = time.time() - save_start_time
# Legacy note (translated to English).
#     logger.info(f"Batch save took: {total_save_time:.2f} seconds")

# def process_single_sample_gpu_extended(params):
#     """
# Legacy note (translated to English).
#     """
#     import time
#     import os
#     import tensorflow as tf
#     import numpy as np
#     import json
#     from datetime import datetime
#     import hashlib
#     import traceback

#     process_start_time = time.time()
# Legacy note (translated to English).
    
#     try:
#         setup_start_time = time.time()
# Legacy note (translated to English).
#         os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
# Legacy note (translated to English).
#         parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
#         if parent_gpu and ',' in parent_gpu:
# Legacy note (translated to English).
#             available_gpus = [gpu.strip() for gpu in parent_gpu.split(',')]
# Legacy note (translated to English).
#             process_id = os.getpid()
#             gpu_index = int(hashlib.md5(str(process_id).encode()).hexdigest(), 16) % len(available_gpus)
#             selected_gpu = available_gpus[gpu_index]
#             os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
#             print(f"[DEBUG] Process for {params['file_id']} (PID: {process_id}) using GPU: {selected_gpu} (from {parent_gpu})")
#         elif parent_gpu:
#             print(f"[DEBUG] Process for {params['file_id']} using parent GPU setting: {parent_gpu}")
#         else:
#             gpu_id = params.get('gpu_id', 0)
#             os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
#             print(f"[DEBUG] Process for {params['file_id']} using GPU: {gpu_id}")
        
# Legacy note (translated to English).
#         gpus = tf.config.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#             except RuntimeError as e:
# Legacy note (translated to English).
# Legacy note (translated to English).
        
# Legacy note (translated to English).
#         start_time = time.time()
#         audio_np = params['audio_data']
#         file_id = params['file_id']
#         cfg = params['cfg']
#         feature_type = params['feature_type']
        
# Legacy note (translated to English).
#         result = {
#             "file_id": file_id,
#             "status": "failed",
#             "data": None,
#             "metadata": None,
#             "error": None
#         }
        
# Legacy note (translated to English).
#         if feature_type == "harmonic":
#             extraction_start_time = time.time()
# Legacy note (translated to English).
#             features_dict = extract_harmonic_distribution_with_controls(audio_np, cfg)
# Legacy note (translated to English).
            
# Legacy note (translated to English).
#             processing_time = time.time() - start_time
#             metadata = {
#                 "file_id": file_id,
#                 "instrument_family": params['instrument_family'],
#                 "instrument_name": params['instrument_name'],
#                 "label": int(params['label']),
#                 "feature_shapes": {
#                     "harmonic_distribution": features_dict['harmonic_distribution'].shape,
#                     "f0_hz": features_dict['f0_hz'].shape,
#                     "loudness_db": features_dict['loudness_db'].shape,
#                     "amps": features_dict['amps'].shape,
#                     "noise_magnitudes": features_dict['noise_magnitudes'].shape,
#                     "f0_confidence": features_dict['f0_confidence'].shape
#                 },
#                 "feature_type": feature_type,
#                 "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "processing_time": processing_time,
#                 "gpu_used": os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
#             }
            
# Legacy note (translated to English).
#             result.update({
#                 "status": "success",
#                 "data": features_dict,
#                 "metadata": metadata
#             })
            
#         else:
#             raise ValueError(f"Unsupported feature type: {feature_type}")
            
#     except Exception as e:
#         result["error"] = f"{str(e)}\n{traceback.format_exc()}"
        
#     finally:
# Legacy note (translated to English).
#         cleanup_tensorflow_memory()
#         total_process_time = time.time() - process_start_time
# Legacy note (translated to English).
        
#     return result

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mp.set_start_method('spawn', force=True)
    
    if getattr(cfg, 'save_extended_features', False):
        logger.info("Using extended feature extraction with memory management")
        extract_nsynth_data_gpu_parallel_extended(
            cfg,
            cfg.split,
            cfg.feature_type,
            cfg.max_samples if hasattr(cfg, 'max_samples') else None,
            cfg.workers_per_gpu if hasattr(cfg, 'workers_per_gpu') else None
        )

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()