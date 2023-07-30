"""Collect utility functions."""

import numpy as np
from scipy.io.wavfile import read
import torch

import score as dscore


def calculate_score(ref_rttm,
                    sys_rttm,
                    p_table=True,
                    collar=0.15,
                    ignore_overlaps=False):
    """calculate DER and JER between `ref_rttm_files` and `sys_rttm_files`.
    each argument contains list of RTTM file names.

    Args:
        ref_rttm_files: list, list that contains reference RTTM file names.
        sys_rttm_files: list, list that contains system RTTM file names.
        p_table: boolean, pretty print scores as table.
        collar: float, collar size in seconds for DER computation.
        ignore_overlaps: boolean, ignore overlaps when computing DER.

    Returns:
        der: float, Diarization error rate in percent.
        jer: float, Jaccard error rate in percent.
    """
    ref_turns, _ = dscore.load_rttms(ref_rttm)
    sys_turns, _ = dscore.load_rttms(sys_rttm)

    uem = dscore.gen_uem(ref_turns, sys_turns)
    ref_turns = dscore.trim_turns(ref_turns, uem)
    sys_turns = dscore.trim_turns(sys_turns, uem)

    ref_turns = dscore.merge_turns(ref_turns)
    sys_turns = dscore.merge_turns(sys_turns)

    dscore.check_for_empty_files(ref_turns, sys_turns, uem)

    file_score, global_score = dscore.score(ref_turns,
                                            sys_turns,
                                            uem,
                                            step=0.01,
                                            jer_min_ref_dur=None,
                                            collar=collar,
                                            ignore_overlaps=ignore_overlaps)

    if p_table:
        dscore.print_table(file_score, global_score, 2, 'simple')

    file_score = file_score[0]

    der = global_score.der
    jer = global_score.jer

    return der, jer


def custom_timeformat(seconds):
    """Convert the seconds to hours, minutes, seconds format"""
    minutes, seconds = divmod(seconds, 60)
    _, minutes = divmod(minutes, 60)

    return '%dmin %.2fsec' % (minutes, seconds)


def print_rt(input_pcm, elapsed_time, sr=8000):
    """This function prints the response time(RT) when processing single file"""
    if isinstance(input_pcm, str):
        sr, audio = read(input_pcm)
    elif isinstance(input_pcm, bytearray):
        audio = np.frombuffer(input_pcm, dtype="int16")
    elif isinstance(input_pcm, bytes):
        audio = np.frombuffer(input_pcm, dtype="int16")
    else:
        raise ValueError("Wrong type: %s" % type(input_pcm))

    datalength = audio.shape[0] / sr
    print("Elapsed_time : {}, data_time : {}, RT : {:.04f}".format(
        custom_timeformat(elapsed_time), custom_timeformat(datalength),
        elapsed_time / datalength))

    return elapsed_time / datalength


##############################################################
##########           VAD RELATED FUNCTIONS          ##########
##############################################################


def parse_vad_segs(vad_segs, max_seg_ms, shift_ms):
    """Resegmentation by slingding windows.

    Args:
        vad_segs: list of 2-tuple of (start_time, end_time).
        max_seg_ms: integer, window size.
        shift_ms: integer, window step.

    Returns:
        starts: list of float, start time.
        ends: list of float, end time.
        seg_ids: list of integer, a index from the input segment
            which result segment belongs to.
    """
    starts, ends, seg_ids = [], [], []

    # chunk_len = float(max_frames)/100
    chunk_len = float(max_seg_ms) / 1000 #1500
    chunk_overlap = float(shift_ms) / 1000 #500
    seg_id = 1

    for seg in vad_segs:
        start = seg[0]
        end = seg[1]

        cur_start = round(start, 2)

        while cur_start + chunk_len < end:
            starts.append(round(cur_start, 2))
            ends.append(round(cur_start + chunk_len, 2))
            seg_ids.append(seg_id)
            cur_start += chunk_overlap
        if cur_start < end:
            starts.append(round(cur_start, 2))
            ends.append(round(end, 2))
            seg_ids.append(seg_id)
        seg_id += 1

    return starts, ends, seg_ids


##############################################################
##########          INPUT FEAT RELATED FCNS         ##########
##############################################################


def stack_pcms(data, max_frames):
    """Sliding window.

    Args:
        data: np.ndarray with shape (n_frames).
        max_frames: integer, window size.

    Returns:
        np.ndarray with shape (n_segments, max_frames).
    """
    datalen = data.shape[0]
    cur_idx = 0
    output = []

    if datalen == max_frames:
        data = np.expand_dims(data, axis=0)
        return data
    while cur_idx < datalen:
        if cur_idx + max_frames > datalen:
            pad_len = (cur_idx + max_frames) - datalen
            cur_data = data[cur_idx:]
            cur_data = np.pad(cur_data, (0, pad_len), 'wrap')
        else:
            cur_data = data[cur_idx:cur_idx + max_frames]

        output.append(cur_data)
        cur_idx += int(max_frames / 2)

    return np.concatenate(output, 0)


def extract_pcms(input_pcm, vad_segments, max_frames, sr=8000):
    """Extract pcm of speech segment from audio.

    Args:
        input_pcm: It can be one of string, bytearray or bytes.
            string, path of audio file.
            bytearray or bytes, data from wavfile.
        vad_segments: list of 2-tuple of (start_time, end_time).
        max_frames: integer, window size.
        sr: integer, sampling rate.

    Returns:
        list of np.ndarray with shape (n_segments, max_frames).
        the shape of each np.ndarray can be different because n_segments
        is determined by the length of segments.
    """
    audios = []
    if isinstance(input_pcm, str):
        sr, audio = read(input_pcm)
    elif isinstance(input_pcm, bytearray):
        audio = np.frombuffer(input_pcm, dtype="int16")
    elif isinstance(input_pcm, bytes):
        audio = np.frombuffer(input_pcm, dtype="int16")
    elif isinstance(input_pcm,
                    np.ndarray) and input_pcm.dtype is np.dtype('int16'):
        audio = input_pcm
    else:
        raise TypeError("Wrong Type: %s" % type(input_pcm))

    starts, ends = [t[0] for t in vad_segments], [t[1] for t in vad_segments]
    parsed = zip(starts, ends)

    audio_len = int(max_frames * (sr / 100)) + int(sr / 100 * 1.5) #24000 + 240
    for seg in parsed:
        s_start = int(seg[0] * sr)
        s_end = int(seg[1] * sr) #if 1s -> s_end = 16000

        audiochunk = audio[s_start:s_end]
        if audiochunk.shape[0] == 0:
            audiochunk = np.zeros(audio_len)
        elif audio.shape[0] < s_end:
            pad_len = s_end - audio.shape[0]
            audiochunk = np.pad(audiochunk, (0, pad_len), 'wrap')

        if audiochunk.shape[0] < audio_len:
            pad_len = audio_len - audiochunk.shape[0]
            audiochunk = np.pad(audiochunk, (0, pad_len), 'wrap')

        audio_batch = stack_pcms(audiochunk, audio_len)
        audios.append(torch.Tensor(audio_batch))
    return audios


##############################################################
###########          DIARIZATION FUNCTIONS          ##########
##############################################################


def merge_speakers(starts, ends, embeddings, cluster_labels, max_seg_ms,
                   shift_ms):
    """Merge consecutive same speaker segments into one.

    Args:
        starts: list contains #segs floats.
        ends: list contains #segs floats.
        embeddings: np.ndarray with shape (#segs, embedding size).
        cluster_labels: np.ndarray with shape (#segs)
        max_seg_ms: integer, window size when extracting speaker embedding(ms).
        shift_ms: integer, window step when extracting speaker embedding(ms).

    Returns:
        merged_starts: list contains #merged_segs floats.
        merged_ends: list contains #merged_segs floats.
        merged_embeddings: np.ndarray with shape (#merged_segs, embedding_size)
        merged_cluster_labels: np.ndarray with shape (#merged_segs)
    """

    prev_label, prev_start, prev_end = -1, -1, -1
    temp_embedding_q = []
    overlap = round(float(max_seg_ms) / 1000 - float(shift_ms) / 1000, 2) # 1.0s = 1.5s - 0.5s

    ret = {key: [] for key in ['starts', 'ends', 'embeddings', 'labels']}

    def _update_ret(start, end, embedding, label):
        start = round(start, 2)
        end = round(end, 2)
        ret['starts'].append(start)
        ret['ends'].append(end)
        ret['embeddings'].append(embedding)
        ret['labels'].append(label)

    sele_tuples = list(zip(starts, ends, cluster_labels, embeddings))
    for start, end, label, embedding in sele_tuples:
        if prev_label >= 0:
            if prev_label != label or start > prev_end:  # Speaker가 다르면,

                if start >= prev_end:  # 다음 segment와 직전 segment가 overlap 되지 않는 경우
                    _update_ret(start=prev_start,
                                end=prev_end,
                                embedding=np.mean(temp_embedding_q, axis=0),
                                label=prev_label)
                    prev_start = start
                else:  # 다음 segment와 직전 segment가 overlap 되는 경우
                    _update_ret(start=prev_start,
                                end=prev_end - overlap / 2,
                                embedding=np.mean(temp_embedding_q, axis=0),
                                label=prev_label)
                    prev_start = start + overlap / 2
                prev_end = end
                temp_embedding_q = [embedding]
            else:  # Speaker가 같으면,
                prev_end = end
                temp_embedding_q.append(embedding)
        else:  # first in 초기 값 설정.
            prev_start = start
            prev_end = end
            temp_embedding_q.append(embedding)

        prev_label = label

    # append the last tuple
    if prev_end > prev_start:
        _update_ret(start=prev_start,
                    end=prev_end,
                    embedding=np.mean(temp_embedding_q, axis=0),
                    label=prev_label)

    return ret['starts'], ret['ends'], ret['embeddings'], ret['labels']


def write_rttm(sel_tuples, out_rttm_file):
    """Write rttm file.

    Args:
        sel_tuples: list of triplet, triplet consists of start_time,
                    end_time or duration and speaker_id.
        out_rttm_file: string, rttm file path.
    """

    file_id = out_rttm_file.split('/')[-1].replace('.rttm', '')
    place_holder = [
        'SPEAKER', file_id, '1', '0', '0', '<NA>', '<NA>', '0', '<NA>', '<NA>'
    ]

    with open(out_rttm_file, 'w') as f_output:
        for tup in sel_tuples:
            place_holder[3] = "%.3f" % tup[0]
            place_holder[4] = "%.3f" % tup[1]
            place_holder[7] = str(tup[2])

            output_string = ' '.join(place_holder)

            f_output.write(output_string + '\n')


def is_valid_type(value, types):
    """Check whether value is one of types.

    Args:
        value: value what we want to check the data type.
        types: list, type whitelist.

    Returns:
        if type of value is one of types, return True.
    """
    return any(isinstance(value, _type) for _type in types)
