import tensorflow as tf
import gc

from ddsp import spectral_ops

def cleanup_tensorflow_memory():
    """clear TensorFlow memory"""
    tf.keras.backend.clear_session()

    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')

    gc.collect()
    spectral_ops.reset_crepe()
        
