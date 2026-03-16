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