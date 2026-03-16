import tensorflow as tf
from ddsp.training import data


class NSynthTfds(data.TfdsProvider):
    """Load NSynth Dataset from TFDS"""
    
    def __init__(self,
                 name='nsynth/full:2.3.3',
                 split='train',
                 data_dir='/home/tensorflow_datasets',
                 sample_rate=16000,
                 frame_rate=250,
                 include_note_labels=True):

        self._include_note_labels = include_note_labels
        super().__init__(name, split, data_dir, sample_rate, frame_rate)
        
        # instrument family 
        self.family_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 
                            'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    
    def get_dataset(self, shuffle=True):
        def preprocess_ex(ex):

            audio = ex['audio']
            
            audio_16k = tf.cast(audio, tf.float32)
            audio_length = tf.shape(audio_16k)[0]
            
            audio_length_float = tf.cast(audio_length, tf.float32)
            sample_rate_float = tf.cast(self.sample_rate, tf.float32)
            frame_rate_float = tf.cast(self.frame_rate, tf.float32)
            
            expected_frames = tf.cast(tf.math.ceil(audio_length_float / (sample_rate_float / frame_rate_float)), tf.int32)
            
            f0_hz = tf.zeros([expected_frames], dtype=tf.float32)
            f0_confidence = tf.zeros([expected_frames], dtype=tf.float32)
            loudness_db = tf.zeros([expected_frames], dtype=tf.float32)
            
            # create output dict
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