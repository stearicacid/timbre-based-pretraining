import tensorflow as tf

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