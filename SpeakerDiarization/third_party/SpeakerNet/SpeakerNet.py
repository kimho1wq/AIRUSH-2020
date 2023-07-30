#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import sys
#FIXME: 여기서 import 오류(다른 경로에서 import 할때) 해결하는 방법은 나중에 @junghoon-jang 님이 도와 주실 수 있을겁니다...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNetSE34L_SA import *
from scipy.io.wavfile import read
from utils import *

__filepath__ = os.path.dirname(os.path.abspath(__file__))

class SpeakerNet(nn.Module):
    def __init__(self, model_path, model_type, max_frames, device='cuda'):
        super(SpeakerNet, self).__init__();
        self.device = device

        self.__S__ = globals()[model_type]().to(self.device)
        self.loadParameters(model_path)
         
        self.__S__.eval()
        self.sr = int(model_type.split('_')[1].replace('k', '')) * 1000      
        #self.max_frames = int(model_type.split('_')[2])
        self.max_frames = max_frames

        self.padd_len = int(self.sr / 100 * 1.5)
        self.audio_len = int(self.max_frames * (self.sr / 100)) + self.padd_len  # Second term is for additional padding

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);
        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");
                name = name.replace("__S__.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

    def get_embedding(self, input_pcm, method='average', shift=1600):
        """Extract speaker embedding.

        Args:
            input_pcm: It can be one of string, bytearray or bytes.
                string, path of audio file.
                bytearray or bytes, data from wavfile.
            method: how to cut input_pcm when input_pcm is longer than
                    the input size of the model.
            shift: int, number of samples to shift, if method is average.

        Returns:
            embeddings, ndarray with shape (batch, embedding_size).
        """
        if type(input_pcm) == str:
            sr, audio = read(input_pcm)
            assert sr == self.sr
        elif type(input_pcm) == bytearray:
            audio = np.frombuffer(input_pcm, dtype="int16")
        elif type(input_pcm)  == bytes:
            audio = np.frombuffer(input_pcm, dtype="int16")
        else:  # numpy array
            audio = input_pcm
            assert audio.dtype == np.int16

        # Make audio 2d tensor with shape (batch, audio_len).
        # (audio_len) --> (1, audio_len)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)

        if method == 'first' or method == 'random':
            audio = pad_or_cut(audio, self.audio_len, method, startpoint = 0)
            with torch.no_grad():
                output = self.__S__.forward(audio.to(self.device), softmax=False)
                output = output.cpu().numpy()
        else:  # method : average
            if audio.shape[1] <= self.audio_len:
                audio = pad_or_cut(audio, self.audio_len, method, startpoint = 0)
                with torch.no_grad():
                    output = self.__S__.forward(audio.to(self.device), softmax=False)
                    output = output.cpu().numpy()
            else:
                emb_list = []
                num_seg = (audio.shape[1] - self.audio_len) // shift
                for i in range(num_seg + 1):
                    audio_seg = pad_or_cut(audio, self.audio_len, method, startpoint = i * shift)
                    with torch.no_grad():
                        output_seg = self.__S__.forward(audio_seg.to(self.device), softmax=False)
                        emb_list.append(output_seg.cpu().numpy())
                output = np.mean(np.array(emb_list), axis=1)
        return output

