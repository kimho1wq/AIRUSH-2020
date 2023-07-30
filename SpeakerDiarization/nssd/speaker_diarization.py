"""Speaker Diarization module."""
#!/usr/bin/python
#-*- coding: utf-8 -*-
import collections

from sklearn.preprocessing import normalize
import numpy as np

from nssd.clustering import Clusterer
from nssd.emb_extractor import EmbeddingExtractor
from nssd.endpoint_detector import EndPointDetector
import nssd.utils as utils

SegmentInfo = collections.namedtuple(
    'SegmentInfo', ['start_time', 'end_time', 'speaker_embedding', 'label'])


class DiarizationInfo:
    """Speaker diarization result of speech audio."""
    def __init__(self, pcm_data, segment_infos):
        """
        Args:
            pcm_data: np.ndarray,
            segment_infos: list of SegmentInfo,
        """
        self.pcm_data = pcm_data
        self.segment_infos = segment_infos

    def to_tuples(self, rttm=False):
        """Export segment_infos as list of tuples.

        Args:
            rttm: boolean, If true, return duration instead of end_time.

        Returns:
            list of 3-tuple of (start_time, end_time or duration, labels).
        """
        if rttm:
            return [(segment.start_time, segment.end_time - segment.start_time,
                     segment.label) for segment in self.segment_infos]
        return [(segment.start_time, segment.end_time, segment.label)
                for segment in self.segment_infos]


class ConversationInfo:
    """Class for describing speaker diarization of converation."""
    def __init__(self, diar_data, diar_eval_data):
        """
        Args:
            diar_data: list of DiarizationInfo,
            diar_eval_data: list of DiarizationEvalInfo,
        """
        self.diar_data = diar_data
        self.diar_eval_data = diar_eval_data

    def export_nssd_data(self):
        """Export all diarization results as a list of list tuples.
        The main use case is for exporting conversation result as a RTTM format.

        Returns:
            rttm_file_names: list of string.
            seg_infos: list of list of speech segment information.
                audio1 - [(seg_info1), ... , (seg_infoN_1)]
                ...
                audioM - [(seg_info1), ... , (seg_infoN_M)]
        """
        rttm_file_names = []
        seg_infos = []
        for idx, diar_info in enumerate(self.diar_data):
            rttm_file_name = '%s.rttm' % str(idx)
            rttm_file_names.append(rttm_file_name)
            seg_infos.append(diar_info.to_tuples(rttm=True))
        return rttm_file_names, seg_infos


def _build_diarization_info(pcm, starts, ends, embeddings, cluster_labels):
    seg_infos = [
        SegmentInfo(start, end, embedding, cluster_label)
        for start, end, embedding, cluster_label in zip(
            starts, ends, embeddings, cluster_labels)
    ]
    return DiarizationInfo(pcm, seg_infos)


def _build_conversation_info(pcms, starts, ends, embeddings, cluster_labels):
    diar_infos = []
    diar_eval_datas = []
    for pcm, start, end, embedding, cluster_label in zip(
            pcms, starts, ends, embeddings, cluster_labels):
        diar_infos.append(
            _build_diarization_info(pcm, start, end, embedding, cluster_label))

    return ConversationInfo(diar_infos, diar_eval_datas)


class SpeakerDiarization():
    """Speaker Diarization module."""
    def __init__(self, **config):

        # Inference parameters
        conf = config.get('inference_config', {})

        self.model_type = conf.get('model_type')
        self.max_frames = conf.get('max_frames')
        self.sr = 16000

        self.model_path = conf.get('model_path', '')
        self.device = conf.get('device', 'cuda')
        self.batch_size = conf.get('batch_size')

        # Diarization parameters
        conf = config.get('diarization_config', {})
        self.max_seg_ms = conf.get('max_seg_ms')
        self.shift_ms = conf.get('shift_ms')
        self.method = conf.get('method', 'ahc')
        self.num_cluster = conf.get('num_cluster', 'None')
        self.normalize = conf.get('normalize', False)

        # Clustering parameters
        clustering_conf = conf.get('clustering_parameters', {})
        clustering_conf['method'] = self.method
        clustering_conf['num_cluster'] = self.num_cluster

        self.clusterer = Clusterer(clustering_conf)

        self.check_parameters()

        # Instantiate speaker embedding extractor and epd detector
        self.embedding_extractor = EmbeddingExtractor(
            model_path=self.model_path,
            model_type=self.model_type,
            batch_size=self.batch_size,
            max_frames=self.max_frames,
            sampling_rate=self.sr,
            device=self.device)

        conf = config.get('epd_config', {})
        self.endpoint_detector = EndPointDetector(conf, sampling_rate=self.sr)

    def clustering(self, embeddings):
        """Run clustering with the extracted embeddings.

        Args:
            embeddings: np.ndarray with shape (n_embeddings, embedding_size),
                embeddings will be clustered.

        Returns:
            labels: np.ndarray with shape (n_embeddings)
        """
        if self.normalize:
            embeddings = normalize(embeddings, axis=1, norm='l2')

        return self.clusterer.predict(embeddings)

    def run(self, input_pcms):
        """Apply speaker diarization on given audios.

        Speaker diarization consists of 4 stages like below:
            1. speech segmentation
            2. extract embedding
            3. clustering
            4. merge spakers

        Args:
            input_pcms: list of string, list of bytearray or list of bytes.
                string, path of audio file.
                bytearray or bytes, data from wavfile.
                Its frame rate should be 8k or 16k.
        Returns:
            ConversationInfo.
        """
        vad_segments = self.get_speech_segments(input_pcms)
        print('len vad_segments',len(vad_segments[0]))
        print('vad_segments',vad_segments)
        embeddings = self.get_embeddings(input_pcms, vad_segments)

        embeddings = np.concatenate(embeddings, 0)
        print('embedding.shape',embeddings.shape)
        cluster_labels = self.clustering(embeddings)
        print('cluster_labels.shape',cluster_labels.shape)
        print('max, min cluster_l',max(cluster_labels),min(cluster_labels))
        print('-----------------------------------------')

        starts = []
        ends = []
        for segment in vad_segments:
            start, end = zip(*segment)
            starts.append(list(start))
            ends.append(list(end))

        print('starts',starts)
        print('ends',ends)
        # Split embeddings and cluster_labels.
        change_point = np.cumsum(
            [len(vad_segment) for vad_segment in vad_segments])
        print('change_point',change_point)
        embeddings = np.split(embeddings, change_point)
        print('len(embeddings)',len(embeddings))
        cluster_labels = np.split(cluster_labels, change_point)
        # merge speakers
        for idx, _ in enumerate(input_pcms):
            starts[idx], ends[idx], embeddings[idx], cluster_labels[
                idx] = utils.merge_speakers(starts[idx], ends[idx],
                                            embeddings[idx],
                                            cluster_labels[idx],
                                            self.max_seg_ms, self.shift_ms)


        return _build_conversation_info(input_pcms, starts, ends, embeddings,
                                        cluster_labels)

    def get_speech_segments(self, input_pcms):
        """Return speech segments of audio.

        Speech segmentation processed in two stages:
            1. Segmentation by VAD module.
            2. Sliding window to make long speech segments fit
               the input tensor shape of SpeakerNet.

        Args:
            input_pcms: list of string, list of bytearray or list of bytes.
                string, path of audio file.
                bytearray or bytes, data from wavfile.
                Its frame rate should be 8k or 16k.
        Returns:
            list of list of 2-tuple of (start_time, end_time).
        """
        ret = []

        segments = [
            self.endpoint_detector.get_epd_result(input_pcm)
            for input_pcm in input_pcms
        ]

        for segment in segments:
            starts, ends, _ = utils.parse_vad_segs(segment, self.max_seg_ms,
                                                   self.shift_ms)
            ret.append(list(zip(starts, ends)))

        return ret

    def get_embeddings(self, input_pcms, vad_segments):
        """Returns speaker embedding of speech segment.

        Args:
            input_pcms: list of string, list of bytearray or list of bytes.
                string, path of audio file.
                bytearray or bytes, data from wavfile.
                Its frame rate should be 8k or 16k.
            vad_segments: list of list of 2-tuple of (start_time, end_time).

        Returns:
            embeddings: list of np.ndarray with shape (n_segments, embedding_size).
        """
        ret = []
        for input_pcm, vad_segment in zip(input_pcms, vad_segments):
            embedding = self.embedding_extractor.get_embeddings(
                input_pcm, vad_segment)
            ret.append(embedding)
        return ret

    def check_parameters(self):
        """This function check the parameters if each of them has
        correct value or not.
        """
        if sum(map(lambda x: 1 if '_' in x else 0, self.model_type)) != 2:
            raise Exception('Model type is unappropriate : ', self.model_type)

        if self.method not in ['ahc']:
            raise Exception('Clustering method is unappropraite : ',
                            self.method)

        if not isinstance(self.normalize, bool):
            raise Exception('Specify "normalize" part correctly as boolean : ',
                            self.normalize)

        if not isinstance(self.num_cluster,
                          int) and self.num_cluster != "None":
            raise Exception('"num_cluster" should be integer or "None" : ',
                            self.num_cluster)

        if self.device not in ['cuda', 'cpu']:
            raise Exception('"device" must be either "cuda" or "cpu"',
                            self.device)
