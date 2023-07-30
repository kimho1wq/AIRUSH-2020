#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy as np
import random

def custom_timeformat(seconds):
    # Convert the seconds to hours, minutes, seconds format
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return '%dmin %.2fsec' % (minutes, seconds)


def pad_or_cut(audio, audio_len, method, startpoint=0):
    """Apply zero-padding or slicing audio along the time axis.

    Args:
        audio: np.ndarray with shape (batch, audio_len).
        audio_len: int, length of result audio.
        method: string, how to select startpoint.
        startpoint: int, startpoint if method is not random.
    Returns:
        torch.Tensor with shape (batch, audio_len).
    """
    axis = 1
    if audio.shape[axis] > audio_len:
        if method == 'random':  # Randomly choose the startpoint of the audio
            startpoint = random.randrange(0, audio.shape[axis] - audio_len)
        audio = audio[:, startpoint:startpoint + audio_len]
    elif audio.shape[axis] < audio_len:
        pad_len = audio_len - audio.shape[axis]
        audio = np.pad(audio, ((0, 0), (0, pad_len)), 'wrap')
    else:
        pass

    audio = torch.Tensor(audio)
    return audio
