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