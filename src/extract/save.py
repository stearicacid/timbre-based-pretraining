import os
import numpy as np
import json

def save_results(results_list, output_dir, all_paths):
    """
    Save harmonic distribution and metadata
    """
    
    for result in enumerate(results_list):
        file_id = result['file_id']
        features_dict = result['data']
        metadata = result['metadata']
        
        # Create paths
        feature_path = os.path.join(output_dir, f"feature_{file_id}.npy")
        f0_path = os.path.join(output_dir, f"f0_{file_id}.npy")
        loudness_path = os.path.join(output_dir, f"loudness_{file_id}.npy")
        amps_path = os.path.join(output_dir, f"amps_{file_id}.npy")
        noise_path = os.path.join(output_dir, f"noise_{file_id}.npy")
        f0_confidence_path = os.path.join(output_dir, f"f0_confidence_{file_id}.npy")
        label_path = os.path.join(output_dir, f"label_{file_id}.npy")
        metadata_path = os.path.join(output_dir, f"metadata_{file_id}.json")

        # Save NumPy arrays
        np.save(feature_path, features_dict['harmonic_distribution'])
        np.save(f0_path, features_dict['f0_hz'])
        np.save(loudness_path, features_dict['loudness_db'])
        np.save(amps_path, features_dict['amps'])
        np.save(noise_path, features_dict['noise_magnitudes'])
        np.save(f0_confidence_path, features_dict['f0_confidence'])
        np.save(label_path, np.array(metadata['label']))


        # Save JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


        # Record paths
        all_paths['feature_paths'].append(feature_path)
        all_paths['f0_paths'].append(f0_path)
        all_paths['loudness_paths'].append(loudness_path)
        all_paths['amps_paths'].append(amps_path)
        all_paths['noise_paths'].append(noise_path)
        all_paths['f0_confidence_paths'].append(f0_confidence_path)
        all_paths['label_paths'].append(label_path)


    