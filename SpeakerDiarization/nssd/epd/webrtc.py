"""webrtc epd."""
import collections

import webrtcvad

class Frame():
    def __init__(self, _bytes, timestamp, duration):
        self.bytes = _bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sr):
    """Split given audio into frames.

    Args:
        frame_duration_ms: integer, frame duration.
        audio: bytearray, data from WAV file.
        sr: integer, sampling rate.

    Returns:
        list of Frame.
    """

    n = int(sr * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sr) / 2.0
    frames = []

    while offset + n < len(audio):
        frames.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    print('frame len',len(frames))
    print('frame[0].timestamp',frames[0].timestamp)
    print('frame[0].duration',frames[0].duration)
    return frames


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad,
                  frames, voice_criteria):
    """
    voice_criteria: criteria for TRIGGERED and NONTRIGGERED state
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms) #10 = 300 / 30
    voice_criteria = voice_criteria * num_padding_frames #7 = 0.7 * 10
    print('num_padding_frames',num_padding_frames)
    print('voice_criteria',voice_criteria)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    s_start, s_end = 0.0, 0.0

    vad_segments = []
    for idx, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate) # Ture or False

        if not triggered: #first in
            ring_buffer.append((frame, is_speech)) #(frame, T or F)
            num_voiced = len([f for f, speech in ring_buffer if speech])

            # If more than voice_criteria of the frames in the ring buffer are voiced,
            # We enter the TRIGGERED state.
            if num_voiced > voice_criteria:
                triggered = True
                s_start = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            # TRIGGERED state: append frames to ring_buffer
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, s in ring_buffer if not s])

            # If more than voice_criteria of the frames in the ring buffer are unvoiced,
            # We enter the NONTRIGGERED state.
            if num_unvoiced > voice_criteria:

                # For smaller margin
                s_end = ring_buffer[int(num_padding_frames / 2)][0].timestamp
                vad_segments.append((s_start, s_end))
                triggered = False
                ring_buffer.clear()

    if triggered:
        if vad_segments and vad_segments[-1][1] == s_start:
            vad_segments[-1] = (vad_segments[-1][0],
                                frame.timestamp + frame.duration)
        else:
            vad_segments.append((s_start, frame.timestamp + frame.duration))

    ## Rounding
    for i in range(len(vad_segments)):
        x, y = vad_segments[i]
        vad_segments[i] = (round(x, 2), round(y, 2))

    return vad_segments

def epd(audio, resolution, sample_rate, voice_criteria):
    """Get positions where a speech segment contains in given audio.

    Args:
        audio: bytearray, data from WAV file.
        resolution: integer, frame duration (ms).
        sample_rate: interger, sampling rate.
        voice_criteria: float, criteria for TRIGGERED and NONTRIGGERED state.
    Returns:
        list of 2-tuple of (start_time, end_time)

    """
    vad = webrtcvad.Vad(2)
    frames = frame_generator(resolution, audio, sample_rate)
    vad_segments = vad_collector(sample_rate, resolution, 400, vad, frames,
                                 voice_criteria)
    print('in epd vad_segments:',vad_segments)
    print('len(epd vad_segments):',len(vad_segments))
    return vad_segments
