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

# --- ログ設定 ---
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
#     """現在のメモリ使用量を取得"""
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     return memory_info.rss / (1024 * 1024 * 1024)  # GB単位

# def cleanup_tensorflow_memory():
#     """TensorFlowのメモリを強制的にクリーンアップ"""
#     try:
#         # TensorFlowのセッションとグラフをクリア
#         tf.keras.backend.clear_session()
        
#         # CUDA関連のメモリをクリア
#         if tf.config.list_physical_devices('GPU'):
#             try:
#                 tf.config.experimental.reset_memory_stats('GPU:0')
#             except Exception as e:
#                 logger.debug(f"GPU memory reset warning: {e}")
        
#         # Python のガベージコレクションを実行
#         gc.collect()
        
#         # DDSPのCREPEモデルをリセット
#         spectral_ops.reset_crepe()
        
#     except Exception as e:
#         logger.warning(f"TensorFlow memory cleanup warning: {e}")

class CustomNSynthTfds(data.TfdsProvider):
    """nsynth/fullデータセット用のカスタムプロバイダー"""
    
    def __init__(self,
                 name='nsynth/full:2.3.3',
                 split='train',
                 data_dir='/mlnas/rin/tensorflow_datasets',
                 sample_rate=16000,
                 frame_rate=250,
                 include_note_labels=True):
        """
        nsynth/fullデータセット用のカスタムプロバイダー
        
        Args:
            name: TFDS dataset name
            split: Dataset split
            data_dir: データセットのディレクトリ
            sample_rate: サンプルレート
            frame_rate: フレームレート
            include_note_labels: 楽器ラベルを含むかどうか
        """
        self._include_note_labels = include_note_labels
        super().__init__(name, split, data_dir, sample_rate, frame_rate)
        
        # 楽器ファミリー名のマッピング
        self.family_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 
                            'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    
    def get_dataset(self, shuffle=True):
        """nsynth/fullデータセットを読み込み、必要な前処理を適用"""
        def preprocess_ex(ex):
            """nsynth/fullデータセット用の前処理関数"""
            # 基本的な音声データを取得
            audio = ex['audio']
            
            # 音声データからf0とloudnessを計算
            audio_16k = tf.cast(audio, tf.float32)
            
            # F0の計算（CREPEを使用）
            # 注意：ここでは一時的にダミー値を設定し、実際の計算はワーカープロセスで行う
            audio_length = tf.shape(audio_16k)[0]
            
            # 型を統一してからceilを計算
            audio_length_float = tf.cast(audio_length, tf.float32)
            sample_rate_float = tf.cast(self.sample_rate, tf.float32)
            frame_rate_float = tf.cast(self.frame_rate, tf.float32)
            
            expected_frames = tf.cast(tf.math.ceil(audio_length_float / (sample_rate_float / frame_rate_float)), tf.int32)
            
            # ダミーのf0とloudnessを作成（実際の計算はワーカープロセスで行う）
            f0_hz = tf.zeros([expected_frames], dtype=tf.float32)
            f0_confidence = tf.zeros([expected_frames], dtype=tf.float32)
            loudness_db = tf.zeros([expected_frames], dtype=tf.float32)
            
            # 出力辞書を構築
            ex_out = {
                'audio': audio_16k,
                'f0_hz': f0_hz,
                'f0_confidence': f0_confidence,
                'loudness_db': loudness_db,
            }
            
            if self._include_note_labels:
                ex_out.update({
                    'pitch': ex['pitch'],
                    'instrument_source': ex['instrument']['source'],
                    'instrument_family': ex['instrument']['family'],
                    'instrument': ex['instrument']['label'],
                })
            
            return ex_out
        
        dataset = super().get_dataset(shuffle)
        dataset = dataset.map(preprocess_ex, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

def extract_harmonic_distribution_with_controls(audio_np, cfg):
    """
    harmonic_distribution に加えて、F0と音量も返すように修正した関数
    """
    func_start_time = time.time()
    print(f"[DEBUG] extract_harmonic_distribution_with_controls: 開始")
    
    sample_rate = getattr(cfg.audio, 'sample_rate', 16000)
    max_length = getattr(cfg.audio, 'max_length', 64000)
    
    if len(audio_np) > max_length:
        audio_np = audio_np[:max_length]
    elif len(audio_np) < max_length:
        audio_np = np.pad(audio_np, (0, max_length - len(audio_np)), 'constant')

    audio_tf = tf.convert_to_tensor([audio_np], dtype=tf.float32)
    
    # 実際のf0とloudnessを計算
    try:
        f0_start_time = time.time()
        spectral_ops.reset_crepe()
        f0_hz_np, f0_confidence_np = spectral_ops.compute_f0(audio_tf[0], frame_rate=250, viterbi=True)
        f0_hz_tf = tf.expand_dims(f0_hz_np, 0)
        loudness_db_tf = spectral_ops.compute_loudness(audio_tf, sample_rate)
        print(f"[DEBUG] F0/Loudness計算時間: {time.time() - f0_start_time:.3f}秒")
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        time_steps = 1000
        f0_hz_tf = tf.ones([1, time_steps], dtype=tf.float32) * 440.0
        loudness_db_tf = tf.ones([1, time_steps], dtype=tf.float32) * -30.0
        f0_confidence_np = tf.ones([time_steps], dtype=tf.float32) * 0.5
    
    cleanup_start_time = time.time()
    cleanup_tensorflow_memory()
    print(f"[DEBUG] メモリクリーンアップ時間: {time.time() - cleanup_start_time:.3f}秒")
    
    try:
        strategy_start_time = time.time()
        # 【修正】複数GPU環境でのGPU分散とシングルGPU戦略を強制
        available_gpus = tf.config.list_physical_devices('GPU')
        parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        if len(available_gpus) > 1 or (parent_gpu and ',' in parent_gpu):
            # 複数GPU環境では最初のGPUのみを使用してMirroredStrategyを回避
            with tf.device('/GPU:0'):
                strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
        else:
            strategy = train_util.get_strategy()
        print(f"[DEBUG] 戦略設定時間: {time.time() - strategy_start_time:.3f}秒")
        
        dataset_start_time = time.time()
        dataset = tf.data.Dataset.from_tensors({
            'audio': audio_tf, 'f0_hz': f0_hz_tf, 'loudness_db': loudness_db_tf
        }).repeat()

        TIME_STEPS = cfg.model.time_steps
        n_samples = audio_tf.shape[1]
        print(f"[DEBUG] データセット作成時間: {time.time() - dataset_start_time:.3f}秒")
        
        model_build_start_time = time.time()
        with strategy.scope():
            preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=TIME_STEPS)
            decoder = decoders.RnnFcDecoder(
                rnn_channels=cfg.model.decoder.rnn_channels, rnn_type=cfg.model.decoder.rnn_type,
                ch=cfg.model.decoder.ch, layers_per_stack=cfg.model.decoder.layers_per_stack,
                input_keys=('ld_scaled', 'f0_scaled'),
                output_splits=(('amps', 1), ('harmonic_distribution', cfg.model.harmonic.n_harmonics),
                               ('noise_magnitudes', cfg.model.harmonic.n_harmonics))
            )
            
            harmonic = ddsp.synths.Harmonic(
                n_samples=n_samples, 
                sample_rate=sample_rate, 
                scale_fn=ddsp.core.exp_sigmoid,
                name='harmonic'
            )
            
            noise = ddsp.synths.FilteredNoise(
                n_samples=n_samples, 
                window_size=getattr(cfg.model.noise, 'window_size', 0),
                initial_bias=getattr(cfg.model.noise, 'initial_bias', -10.0),
                name='noise'
            )
            
            add = ddsp.processors.Add(name='add')

            dag = [
                (harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
                (noise, ['noise_magnitudes']),
                (add, ['noise/signal', 'harmonic/signal'])
            ]
            
            processor_group = ddsp.processors.ProcessorGroup(dag=dag, name='processor_group')
            spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1', mag_weight=1.0, logmag_weight=1.0)
            model = models.Autoencoder(preprocessor=preprocessor, encoder=None, decoder=decoder, processor_group=processor_group, losses=[spectral_loss])
            trainer = ddsp.training.trainers.Trainer(model, strategy, learning_rate=cfg.model.training.learning_rate)
        print(f"[DEBUG] モデル構築時間: {time.time() - model_build_start_time:.3f}秒")

        dataset_prep_start_time = time.time()
        dataset = trainer.distribute_dataset(dataset)
        dataset_iter = iter(dataset)
        print(f"[DEBUG] データセット準備時間: {time.time() - dataset_prep_start_time:.3f}秒")
        
        training_start_time = time.time()
        n_steps = cfg.model.training.n_steps
        print(f"[DEBUG] トレーニング開始 - ステップ数: {n_steps}")
        for step in range(n_steps):
            step_start = time.time()
            trainer.train_step(dataset_iter)
            if step % max(1, n_steps // 10) == 0:  # 10%ごとに進捗を出力
                print(f"[DEBUG] ステップ {step}/{n_steps} 完了 (時間: {time.time() - step_start:.3f}秒)")
        print(f"[DEBUG] トレーニング時間: {time.time() - training_start_time:.3f}秒")
        
        inference_start_time = time.time()
        batch = next(dataset_iter)
        controls = model(batch)
        
        harmonic_distribution_np = controls['harmonic_distribution'][0].numpy()
        amps_np = controls['amps'][0].numpy()
        noise_magnitudes_np = controls['noise_magnitudes'][0].numpy()
        print(f"[DEBUG] 推論時間: {time.time() - inference_start_time:.3f}秒")
        
        # NumPy配列への変換を修正
        # f0_hz_np は既に NumPy 配列なので、さらに .numpy() は不要
        if isinstance(f0_hz_np, tf.Tensor):
            f0_hz_original = f0_hz_np.numpy()
        else:
            f0_hz_original = f0_hz_np
            
        loudness_db_original = loudness_db_tf[0].numpy()
        
        # f0_confidence_np の型をチェック
        if isinstance(f0_confidence_np, tf.Tensor):
            f0_confidence_original = f0_confidence_np.numpy()
        else:
            f0_confidence_original = f0_confidence_np
        
        total_time = time.time() - func_start_time
        print(f"[DEBUG] extract_harmonic_distribution_with_controls: 合計時間 {total_time:.3f}秒")
        
        return {
            'harmonic_distribution': harmonic_distribution_np,
            'amps': amps_np,
            'noise_magnitudes': noise_magnitudes_np,
            'f0_hz': f0_hz_original,
            'loudness_db': loudness_db_original,
            'f0_confidence': f0_confidence_original
        }
        
    finally:
        # cleanup_tensorflow_memory() # ← finallyブロック内の呼び出しも不要
        pass # finallyは残しても良い

def extract_nsynth_data_gpu_parallel_extended(cfg, split="train", feature_type="harmonic", max_samples=None, workers_per_gpu=None):
    """
    拡張版メモリ管理機能付きGPU並列処理関数（修正版）
    """
    print(f"[DEBUG] extract_nsynth_data_gpu_parallel_extended 開始")
    
    output_dir = os.path.join(cfg.output_dir, split)
    os.makedirs(output_dir, exist_ok=True)
    
    start_time_total = time.time()
    logger.info(f"Starting EXTENDED GPU-PARALLEL NSynth processing ({split} split)")
    
    # 【修正】親プロセスのGPU設定を取得
    parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_gpu:
        logger.info(f"Using parent GPU setting: {parent_gpu}")
        # 親プロセスで設定されているGPUを使用
        gpu_ids = [int(gpu_id.strip()) for gpu_id in parent_gpu.split(',')]
        num_physical_gpus = len(gpu_ids)
    else:
        # 親プロセスでGPU設定がない場合は物理GPUを検出
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

    # データセットの準備
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
    
    # タスクリストを事前に準備
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
                # 【修正】親プロセスのGPU設定がある場合は削除、ない場合のみ設定
                'gpu_id': gpu_ids[sample_count % num_physical_gpus] if not parent_gpu else None
            }
            
            tasks_to_process.append(params)
            sample_count += 1
    
    print(f"[DEBUG] タスク準備時間: {time.time() - task_prep_start_time:.3f}秒")
    logger.info(f"Prepared {len(tasks_to_process)} tasks for processing")
    
    all_paths = {
        'feature_paths': [], 'f0_paths': [], 'loudness_paths': [],
        'amps_paths': [], 'noise_paths': [], 'f0_confidence_paths': [], 'label_paths': []
    }
    
    completed_tasks = 0
    failed_tasks = 0
    
    # 【変更】バッチ保存用の設定
    SAVE_BATCH_SIZE = workers_per_gpu * num_physical_gpus # ワーカーの総数ごとに保存
    results_batch = []

    try:
        pool_start_time = time.time()
        with mp.Pool(processes=total_workers, maxtasksperchild=1) as pool:
            logger.info("Starting parallel processing with ProcessPool...")
            print(f"[DEBUG] プールの開始時間: {time.time() - pool_start_time:.3f}秒")

            with tqdm(total=len(tasks_to_process), desc="Processing samples") as pbar:
                for result in pool.imap_unordered(process_single_sample_gpu_extended, tasks_to_process, chunksize=1):
                    if result and result["status"] == "success":
                        results_batch.append(result) # 結果をバッチに追加
                        completed_tasks += 1
                    else:
                        logger.error(f"Task failed: {result.get('file_id', 'unknown')}: {result.get('error', 'unknown error')}")
                        failed_tasks += 1
                    pbar.update(1)

                    # 【追加】バッチサイズに達したら保存
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
        # 【追加】ループ終了後、残りの結果を保存
        if results_batch:
            logger.info(f"Saving final batch of {len(results_batch)} files...")
            save_results(results_batch, output_dir, all_paths)
            results_batch.clear()

    processing_time = time.time() - start_time_total
    logger.info(f"EXTENDED processing completed: {completed_tasks} successful, {failed_tasks} failed")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    
    # 既存ファイルのパスも追加
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
#     結果のバッチをファイルに保存するヘルパー関数
#     """
#     save_start_time = time.time()
#     print(f"[DEBUG] バッチ保存開始: {len(results_list)}件のファイル")
    
#     for i, result in enumerate(results_list):
#         file_save_start = time.time()
#         file_id = result['file_id']
#         features_dict = result['data']
#         metadata = result['metadata']
        
#         # パスを生成
#         path_gen_start = time.time()
#         feature_path = os.path.join(output_dir, f"feature_{file_id}.npy")
#         f0_path = os.path.join(output_dir, f"f0_{file_id}.npy")
#         loudness_path = os.path.join(output_dir, f"loudness_{file_id}.npy")
#         amps_path = os.path.join(output_dir, f"amps_{file_id}.npy")
#         noise_path = os.path.join(output_dir, f"noise_{file_id}.npy")
#         f0_confidence_path = os.path.join(output_dir, f"f0_confidence_{file_id}.npy")
#         label_path = os.path.join(output_dir, f"label_{file_id}.npy")
#         metadata_path = os.path.join(output_dir, f"metadata_{file_id}.json")
#         print(f"[DEBUG] {file_id}: パス生成時間: {time.time() - path_gen_start:.3f}秒")

#         # NumPy配列の保存
#         numpy_save_start = time.time()
#         np.save(feature_path, features_dict['harmonic_distribution'])
#         np.save(f0_path, features_dict['f0_hz'])
#         np.save(loudness_path, features_dict['loudness_db'])
#         np.save(amps_path, features_dict['amps'])
#         np.save(noise_path, features_dict['noise_magnitudes'])
#         np.save(f0_confidence_path, features_dict['f0_confidence'])
#         np.save(label_path, np.array(metadata['label']))
#         numpy_save_time = time.time() - numpy_save_start
#         print(f"[DEBUG] {file_id}: NumPy配列保存時間: {numpy_save_time:.3f}秒")
        
#         # JSON保存
#         json_save_start = time.time()
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
#         json_save_time = time.time() - json_save_start
#         print(f"[DEBUG] {file_id}: JSON保存時間: {json_save_time:.3f}秒")

#         # パスを記録
#         path_record_start = time.time()
#         all_paths['feature_paths'].append(feature_path)
#         all_paths['f0_paths'].append(f0_path)
#         all_paths['loudness_paths'].append(loudness_path)
#         all_paths['amps_paths'].append(amps_path)
#         all_paths['noise_paths'].append(noise_path)
#         all_paths['f0_confidence_paths'].append(f0_confidence_path)
#         all_paths['label_paths'].append(label_path)
#         path_record_time = time.time() - path_record_start
#         print(f"[DEBUG] {file_id}: パス記録時間: {path_record_time:.3f}秒")
        
#         file_total_time = time.time() - file_save_start
#         print(f"[DEBUG] {file_id}: 単一ファイル保存合計時間: {file_total_time:.3f}秒 (NumPy: {numpy_save_time:.3f}s, JSON: {json_save_time:.3f}s)")
        
#         # 進捗表示（10件ごと）
#         if (i + 1) % 10 == 0 or (i + 1) == len(results_list):
#             print(f"[DEBUG] バッチ保存進捗: {i + 1}/{len(results_list)} 完了")
    
#     total_save_time = time.time() - save_start_time
#     print(f"[DEBUG] バッチ保存完了: 合計時間 {total_save_time:.2f}秒, 平均 {total_save_time/len(results_list):.3f}秒/ファイル")
#     logger.info(f"Batch save took: {total_save_time:.2f} seconds")

def process_single_sample_gpu_extended(params):
    """
    GPU拡張版の単一サンプル処理関数（ファイル保存を行わず、データを返す）
    """
    import time
    import os
    import tensorflow as tf
    import numpy as np
    import json
    from datetime import datetime
    import hashlib
    import traceback

    process_start_time = time.time()
    print(f"[DEBUG] プロセス {params['file_id']} 開始 (PID: {os.getpid()})")
    
    try:
        setup_start_time = time.time()
        # 【修正】環境変数を設定してMirroredStrategyを無効化
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # 【修正】プロセスIDに基づいてGPUを分散
        parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if parent_gpu and ',' in parent_gpu:
            # 複数GPU設定の場合、プロセスIDに基づいてGPUを選択
            available_gpus = [gpu.strip() for gpu in parent_gpu.split(',')]
            # プロセスIDのハッシュを使ってGPUを分散
            process_id = os.getpid()
            gpu_index = int(hashlib.md5(str(process_id).encode()).hexdigest(), 16) % len(available_gpus)
            selected_gpu = available_gpus[gpu_index]
            os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
            print(f"[DEBUG] Process for {params['file_id']} (PID: {process_id}) using GPU: {selected_gpu} (from {parent_gpu})")
        elif parent_gpu:
            print(f"[DEBUG] Process for {params['file_id']} using parent GPU setting: {parent_gpu}")
        else:
            gpu_id = params.get('gpu_id', 0)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[DEBUG] Process for {params['file_id']} using GPU: {gpu_id}")
        
        # GPUメモリ設定
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                pass  # 初期化済みの場合は無視
        print(f"[DEBUG] GPU設定時間: {time.time() - setup_start_time:.3f}秒")
        
        # 処理開始
        start_time = time.time()
        audio_np = params['audio_data']
        file_id = params['file_id']
        cfg = params['cfg']
        feature_type = params['feature_type']
        
        # 【変更】結果の構造を変更
        result = {
            "file_id": file_id,
            "status": "failed",
            "data": None,
            "metadata": None,
            "error": None
        }
        
        # 特徴量抽出
        if feature_type == "harmonic":
            extraction_start_time = time.time()
            print(f"[DEBUG] {file_id}: 特徴量抽出開始")
            features_dict = extract_harmonic_distribution_with_controls(audio_np, cfg)
            print(f"[DEBUG] {file_id}: 特徴量抽出時間: {time.time() - extraction_start_time:.3f}秒")
            
            # メタデータ作成（内容は変更なし）
            processing_time = time.time() - start_time
            metadata = {
                "file_id": file_id,
                "instrument_family": params['instrument_family'],
                "instrument_name": params['instrument_name'],
                "label": int(params['label']),
                "feature_shapes": {
                    "harmonic_distribution": features_dict['harmonic_distribution'].shape,
                    "f0_hz": features_dict['f0_hz'].shape,
                    "loudness_db": features_dict['loudness_db'].shape,
                    "amps": features_dict['amps'].shape,
                    "noise_magnitudes": features_dict['noise_magnitudes'].shape,
                    "f0_confidence": features_dict['f0_confidence'].shape
                },
                "feature_type": feature_type,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": processing_time,
                "gpu_used": os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
            }
            
            # 【変更】結果にデータとメタデータを格納
            result.update({
                "status": "success",
                "data": features_dict,
                "metadata": metadata
            })
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
            
    except Exception as e:
        result["error"] = f"{str(e)}\n{traceback.format_exc()}"
        
    finally:
        # メモリクリーンアップ
        cleanup_tensorflow_memory()
        total_process_time = time.time() - process_start_time
        print(f"[DEBUG] プロセス {params['file_id']} 完了: 合計時間 {total_process_time:.3f}秒")
        
    return result

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