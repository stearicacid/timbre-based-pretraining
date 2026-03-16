import os
import tensorflow as tf
from datetime import datetime
import hashlib

from utils.utils import cleanup_tensorflow_memory
from extract.features import extract_harmonic_distribution

def process_single_sample(params):
    """
    GPU parallel processing function 
    """

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    parent_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_gpu and ',' in parent_gpu:
        available_gpus = [gpu.strip() for gpu in parent_gpu.split(',')]
        process_id = os.getpid()
        gpu_index = int(hashlib.md5(str(process_id).encode()).hexdigest(), 16) % len(available_gpus)
        selected_gpu = available_gpus[gpu_index]
        os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
    elif parent_gpu:
        print(f"[DEBUG] Process for {params['file_id']} using parent GPU setting: {parent_gpu}")
    else:
        gpu_id = params.get('gpu_id', 0)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[DEBUG] Process for {params['file_id']} using GPU: {gpu_id}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass  

    

    audio_np = params['audio_data']
    file_id = params['file_id']
    cfg = params['cfg']
    feature_type = params['feature_type']

    result = {
        "file_id": file_id,
        "status": "failed",
        "data": None,
        "metadata": None,
        "error": None
    }
    
    # feature extraction
    if feature_type == "harmonic":
        features_dict = extract_harmonic_distribution(audio_np, cfg)
        
        # create metadata
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
            "gpu_used": os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
        }
        
        result.update({
            "status": "success",
            "data": features_dict,
            "metadata": metadata
        })
        
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    

    cleanup_tensorflow_memory()
        
    return result