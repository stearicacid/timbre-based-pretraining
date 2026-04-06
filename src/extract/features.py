import os
import numpy as np
import tensorflow as tf

import ddsp
from ddsp.training import models, preprocessing, decoders
from ddsp import spectral_ops
from ddsp.training import train_util

from utils.utils import cleanup_tensorflow_memory

import logging
logger = logging.getLogger(__name__)

def extract_harmonic_distribution(audio_np, cfg):
    """
    extract harmonic distribution from audio using DDSP model with controls (f0, loudness) 
    """
    
    sample_rate = getattr(cfg.audio, 'sample_rate', 16000)
    max_length = getattr(cfg.audio, 'max_length', 64000)

    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim != 1:
        raise ValueError(f"audio_np must be 1D, got shape={audio_np.shape}")
    
    if len(audio_np) > max_length:
        audio_np = audio_np[:max_length]
    elif len(audio_np) < max_length:
        audio_np = np.pad(audio_np, (0, max_length - len(audio_np)), 'constant')

    audio_tf = tf.convert_to_tensor([audio_np], dtype=tf.float32)
    
    try:
        spectral_ops.reset_crepe()
        f0_hz_np, f0_confidence_np = spectral_ops.compute_f0(
            audio_tf[0],
            frame_rate=250,
            viterbi=True,
        )
        loudness_db_tf = spectral_ops.compute_loudness(audio_tf, sample_rate)
        f0_hz_tf = tf.convert_to_tensor(f0_hz_np[None, :], dtype=tf.float32)

    except Exception as e:
        logger.warning(f"Failed to compute f0 and loudness using CREPE. Using default values: {e}")
        time_steps = cfg.model.time_steps
        f0_hz_np = np.full((time_steps,), 440.0, dtype=np.float32)
        f0_confidence_np = np.full((time_steps,), 0.5, dtype=np.float32)
        loudness_db_np = np.full((time_steps,), -30.0, dtype=np.float32)

        f0_hz_tf = tf.convert_to_tensor(f0_hz_np[None, :], dtype=tf.float32)
        loudness_db_tf = tf.convert_to_tensor(loudness_db_np[None, :], dtype=tf.float32)

    cleanup_tensorflow_memory()

    available_gpus = tf.config.list_physical_devices('GPU')
    parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if len(available_gpus) > 1 or (parent_gpu and ',' in parent_gpu):
        with tf.device('/GPU:0'):
            strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
    else:
        strategy = train_util.get_strategy()

    dataset = tf.data.Dataset.from_tensors({
        'audio': audio_tf, 'f0_hz': f0_hz_tf, 'loudness_db': loudness_db_tf
    }).repeat()

    time_steps = cfg.model.time_steps
    n_samples = audio_tf.shape[1]

    with strategy.scope():
        preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=time_steps)
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

    dataset = trainer.distribute_dataset(dataset)
    dataset_iter = iter(dataset)

    trainer.train_step(dataset_iter)

    batch = next(dataset_iter)
    controls = model(batch)
    
    harmonic_distribution_np = controls['harmonic_distribution'][0].numpy()
    amps_np = controls['amps'][0].numpy()
    noise_magnitudes_np = controls['noise_magnitudes'][0].numpy()
        
    loudness_db_original = loudness_db_tf[0].numpy()
    f0_hz_np = np.asarray(f0_hz_np, dtype=np.float32)[:time_steps]
    loudness_db_original = loudness_db_tf[0].numpy().astype(np.float32, copy=False)[:time_steps]
    f0_confidence_np = np.asarray(f0_confidence_np, dtype=np.float32)[:time_steps]
    
    return {
        'harmonic_distribution': harmonic_distribution_np,
        'amps': amps_np,
        'noise_magnitudes': noise_magnitudes_np,
        'f0_hz': f0_hz_np,
        'loudness_db': loudness_db_original,
        'f0_confidence': f0_confidence_np
    }
