#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import argparse
from SpeakerNet import SpeakerNet

def main(args):
    test_file = './sample/test.wav'

    speakernet = SpeakerNet(model_path=args.model_path, model_type=args.model_type, device=args.device)
    embedding = speakernet.get_embedding(test_file, method=args.method, shift=args.shift) # This is the speaker embedding of the corresponding wavfile.
    print(embedding.shape)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Speaker Embedding Extractor")
    parser.add_argument('--model_type', dest='model_type', default='WeiCheng_16k_100', help='Modeltype for Speaker Embedding extractor')
    parser.add_argument('--model_path', dest='model_path', default='models/weights/16k/WeiCheng_16k_100.model', help='path where model weight file exists')
    parser.add_argument('--method', dest='method', default='first', choices=['first', 'random', 'average'], help='Method to extract embeddings')
    parser.add_argument('--shift', dest='shift', type=int, default=1600, help='Number of samples to shift when method is average')
    parser.add_argument('--device', dest='device', default='cpu', choices=['cpu', 'cuda'], help='device to use during inference')
    args = parser.parse_args()
    main(args)
