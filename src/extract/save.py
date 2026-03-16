import os
import numpy as np
import time
import json

def save_results(results_list, output_dir, all_paths):
    """
    結果のバッチをファイルに保存するヘルパー関数
    """
    save_start_time = time.time()
    print(f"[DEBUG] バッチ保存開始: {len(results_list)}件のファイル")
    
    for i, result in enumerate(results_list):
        file_save_start = time.time()
        file_id = result['file_id']
        features_dict = result['data']
        metadata = result['metadata']
        
        # パスを生成
        path_gen_start = time.time()
        feature_path = os.path.join(output_dir, f"feature_{file_id}.npy")
        f0_path = os.path.join(output_dir, f"f0_{file_id}.npy")
        loudness_path = os.path.join(output_dir, f"loudness_{file_id}.npy")
        amps_path = os.path.join(output_dir, f"amps_{file_id}.npy")
        noise_path = os.path.join(output_dir, f"noise_{file_id}.npy")
        f0_confidence_path = os.path.join(output_dir, f"f0_confidence_{file_id}.npy")
        label_path = os.path.join(output_dir, f"label_{file_id}.npy")
        metadata_path = os.path.join(output_dir, f"metadata_{file_id}.json")
        print(f"[DEBUG] {file_id}: パス生成時間: {time.time() - path_gen_start:.3f}秒")

        # NumPy配列の保存
        numpy_save_start = time.time()
        np.save(feature_path, features_dict['harmonic_distribution'])
        np.save(f0_path, features_dict['f0_hz'])
        np.save(loudness_path, features_dict['loudness_db'])
        np.save(amps_path, features_dict['amps'])
        np.save(noise_path, features_dict['noise_magnitudes'])
        np.save(f0_confidence_path, features_dict['f0_confidence'])
        np.save(label_path, np.array(metadata['label']))
        numpy_save_time = time.time() - numpy_save_start
        print(f"[DEBUG] {file_id}: NumPy配列保存時間: {numpy_save_time:.3f}秒")
        
        # JSON保存
        json_save_start = time.time()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        json_save_time = time.time() - json_save_start
        print(f"[DEBUG] {file_id}: JSON保存時間: {json_save_time:.3f}秒")

        # パスを記録
        path_record_start = time.time()
        all_paths['feature_paths'].append(feature_path)
        all_paths['f0_paths'].append(f0_path)
        all_paths['loudness_paths'].append(loudness_path)
        all_paths['amps_paths'].append(amps_path)
        all_paths['noise_paths'].append(noise_path)
        all_paths['f0_confidence_paths'].append(f0_confidence_path)
        all_paths['label_paths'].append(label_path)
        path_record_time = time.time() - path_record_start
        print(f"[DEBUG] {file_id}: パス記録時間: {path_record_time:.3f}秒")
        
        file_total_time = time.time() - file_save_start
        print(f"[DEBUG] {file_id}: 単一ファイル保存合計時間: {file_total_time:.3f}秒 (NumPy: {numpy_save_time:.3f}s, JSON: {json_save_time:.3f}s)")
        
        # 進捗表示（10件ごと）
        if (i + 1) % 10 == 0 or (i + 1) == len(results_list):
            print(f"[DEBUG] バッチ保存進捗: {i + 1}/{len(results_list)} 完了")
    
    total_save_time = time.time() - save_start_time
    print(f"[DEBUG] バッチ保存完了: 合計時間 {total_save_time:.2f}秒, 平均 {total_save_time/len(results_list):.3f}秒/ファイル")
    # logger.info(f"Batch save took: {total_save_time:.2f} seconds")