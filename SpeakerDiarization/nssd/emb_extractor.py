"""Embedding extractor."""
import numpy as np
import torch

import nssd.utils as utils
from SpeakerNet import SpeakerNet

def split_segment(segment, batch_size):
    """Split the speech segment into small segments to be consumable
    by SpeakerNet.

    Args:
        segment: np.ndarray with shape (n_windows, window_size).
        batch_size: integer, batch size.

    Returns:
        np.ndarray with shape (batch_size, window_size).
    """
    n_windows = len(segment)
    print('n_windows, batch_size:',n_windows, batch_size)
    for cur_idx in range(0, n_windows, batch_size):
        if cur_idx + batch_size >= n_windows:
            yield segment[cur_idx:]
        else:
            yield segment[cur_idx:cur_idx+batch_size]

class EmbeddingExtractor():
    """Extract feature vector of speech segment.

    Args:
        model_path: string, weight file path of SpeakerNet.
        model_type: string, SpeakerNet model type.
        batch_size: integer, batch size.
        max_frames: integer, window size.
        sampling_rate: integer, sampling rate of audio.
        device: string, 'cuda' for running with GPU,
            'cpu' for running with CPU.

    Example of using EmbeddingExtractor:
        # Instantiation.
        embedding_extractor = EmbeddingExtractor(
                                model_path='<weight_dir>/ResNetSE_16k_150.model',
                                model_type='ResNetSE_16k_150',
                                batch_size=512,
                                max_frames=150,
                                sampling_rate=16000,
                                device='cuda')

        input_pcm = '<audio_dir>/audio.wav'
        vad_segment = [(0.14, 2.33), (3.12, 5.77), (10.01, 15.22)]
        embeddings = embedding_extractor(input_pcm, vad_segment)
    """
    def __init__(self,
                 model_path,
                 model_type,
                 batch_size,
                 max_frames,
                 sampling_rate=16000,
                 device='cuda'):
        assert model_path != ''
        self.max_frames = max_frames
        self.device = torch.device(device)
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size

        self.speakernet = SpeakerNet(model_path=model_path,
                                     model_type=model_type,
                                     max_frames=max_frames,
                                     device=self.device)

    def get_embeddings(self, input_pcm, vad_segments):
        """Get embeddings of audio segments.

        Args:
            input_pcm: It can be one of string, bytearray or bytes.
                string, path of audio file.
                bytearray or bytes, data from wavfile.
            vad_segments: list of tuple, which is (start_time, end_time).

        Returns:
            embeddings: np.ndarray with shape (the number of segments, embedding_size)
        """
        embeddings = []
        max_frames = self.max_frames
        batch_size = self.batch_size

        # Chop the pcm in to segments from input_pcm
        pcm_segs = utils.extract_pcms(input_pcm,
                                      vad_segments,
                                      max_frames,
                                      sr=self.sampling_rate)
        pcm_segs = [np.concatenate(pcm_segs, axis=0)]
        # input segment corresponding to each vad segment
        for idx, seg in enumerate(pcm_segs):
            seg = seg.astype(np.short)
            cur_segment = []

            batches = split_segment(seg, batch_size)
            for jj, batch in enumerate(batches):
                with torch.no_grad():
                    output = self.speakernet.get_embedding(batch)
                cur_segment.append(output)

            cur_segment = np.concatenate(cur_segment, 0)
            embeddings.append(cur_segment)

        return np.concatenate(embeddings, 0)
