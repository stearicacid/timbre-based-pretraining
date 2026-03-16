import os
import numpy as np
import tensorflow as tf
import time

import ddsp
from ddsp.training import models, preprocessing, decoders
from ddsp import spectral_ops
from ddsp.training import train_util

from ..utils.utils import cleanup_tensorflow_memory

def extract_harmonic_distribution(audio_np, cfg):
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