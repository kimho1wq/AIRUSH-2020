#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
from scipy.io import wavfile
from queue import Queue
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms
from scipy import signal

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class wav_split(Dataset):
    def __init__(self, dataset_file_name, max_frames, train_path, batch_size, noise_aug = False):
        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames
        self.noise_aug = noise_aug
        # self.instancenorm   = nn.InstanceNorm1d(40)
        self.data_dict = {}
        self.data_list = []
        self.nFiles = 0
        self.batch_size = batch_size

        self.noisetypes = ['noise','speech','music']

        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.noiselist = {}
      
        augment_files   = glob.glob('/home1/irteam/db/musan/*/*/*/*.wav');
     
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        ### Read Training Files...
        self.spk_dic = {}
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;
                
                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(train_path,data[1]);
                if speaker_name not in self.spk_dic:
                    self.spk_dic[speaker_name] = []

                self.data_list.append(filename)
                self.spk_dic[speaker_name].append(filename)
        
    

    def make_spk_list(self):
        self.spk_list = []

        while len(self.spk_list) < self.__len__():
            spk_list_tmp = list(self.spk_dic.keys())
            numpy.random.shuffle(spk_list_tmp)
            self.spk_list.extend(spk_list_tmp)

                
    def __getitem__(self, index):
        fns = numpy.random.choice(self.spk_dic[self.spk_list[index]],size = 2, replace = False)
        
        audio = []
        audio.append(loadWAV(fns[0], self.max_frames, evalmode=False).astype(numpy.float)[0])
        audio.append(loadWAV(fns[1], self.max_frames, evalmode=False).astype(numpy.float)[0])
      
        
        if self.noise_aug:
            augment_profiles = []
            audio_aug = []
            for ii in range(len(audio)):
                ## additive noise profile
                noisecat    = random.choice(self.noisetypes)
                noisefile   = random.choice(self.noiselist[noisecat].copy())
                snr = [random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])]
                augment_profiles.append({'add_noise': noisefile, 'add_snr': snr})
                
                audio_aug.append(self.augment_wav(audio[0],augment_profiles[0]))
                audio_aug.append(self.augment_wav(audio[1],augment_profiles[1]))
            
            audio = numpy.concatenate(audio_aug,axis=0)
        else:
            audio = numpy.stack(audio,axis=0)
        
        audio = torch.FloatTensor(audio)
             
        return audio

    def __len__(self):
        return len(self.data_list)


    def augment_wav(self,audio,augment):

        noiseaudio  = loadWAV(augment['add_noise'], self.max_frames, evalmode=False).astype(numpy.float)

        noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        noise = numpy.sqrt(10 ** ((clean_db - noise_db - augment['add_snr']) / 10)) * noiseaudio
        audio = audio + noise
        
        return audio


def round_down(num, divisor):
    return num - (num%divisor)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0)

    #feat = torch.FloatTensor(feat)

    return feat;


def get_data_loader(dataset_file_name, batch_size, max_frames, nDataLoaderThread, train_path,  **kwargs):
    
    train_dataset = wav_split(dataset_file_name, max_frames, train_path, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nDataLoaderThread,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    
    return train_loader
