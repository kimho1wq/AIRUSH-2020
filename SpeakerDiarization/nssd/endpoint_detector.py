"""End point detector."""
import contextlib
import wave

import nssd.epd.webrtc as webrtc

def read_wave(path):
    """Read WAV file.

    Args:
        path: string, path of WAV format file.

    Returns:
        pcm_data: bytes, raw wave data.
        sample_rate: integer, sampling frequency.
    """

    with contextlib.closing(wave.open(path, 'rb')) as wave_file:
        num_channels = wave_file.getnchannels()
        assert num_channels == 1
        sample_width = wave_file.getsampwidth()
        assert sample_width == 2
        sample_rate = wave_file.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wave_file.readframes(wave_file.getnframes())

        return pcm_data, sample_rate


class EndPointDetector():
    def __init__(self, config, sampling_rate=8000):
        self.mode = config.get('epd_mode', 'webrtc')
        self.resolution = config.get('resolution', 30)
        self.voice_criteria = config.get('voice_criteria', 0.7)
        self.sampling_rate = sampling_rate

    def get_epd_result(self, input_pcm):
        """Get positions where a speech segment contains in given audio.

        Args:
            input_pcm: string, bytearray or bytes.
                string, path of audio file.
                bytearray or bytes, data from WAV file.
                Its frame rate should be 8k or 16k.
        Returns:
            list of 2-tuple of (start_time, end_time)
        """
        vad_segments = []

        voice_criteria = self.voice_criteria
        resolution = self.resolution
        mode = self.mode
        sampling_rate = self.sampling_rate

        if isinstance(input_pcm, str):
            audio, sampling_rate = read_wave(input_pcm)
        elif isinstance(input_pcm, bytearray):
            audio = input_pcm
        elif isinstance(input_pcm, bytes):
            audio = bytearray(input_pcm)
        else:
            raise TypeError("Unsupported input pcm: %s" % type(input_pcm))

        if mode == 'webrtc':
            print('sampling_rate',sampling_rate)
            vad_segments = webrtc.epd(audio, resolution, sampling_rate, voice_criteria)
        else:
            raise ValueError("Unsupported EPD mode: %s" % mode)

        return vad_segments
