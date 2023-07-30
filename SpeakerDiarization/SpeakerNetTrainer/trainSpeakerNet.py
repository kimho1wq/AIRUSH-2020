#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
from tuneThreshold import tuneThresholdfromScore
from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader

try:
    import nsml
    from nsml import DATASET_PATH
except:
    DATASET_PATH = ''
    pass;

parser = argparse.ArgumentParser(description = "SpeakerNet");

## Data loader
parser.add_argument('--max_frames', type=int, default=200,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=70, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="angleproto",    help='Loss function');
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');


## Training and test data
parser.add_argument('--train_list', type=str, default="",   help='Train list');
parser.add_argument('--test_list',  type=str, default="",   help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--initial_model',  type=str, default="./models/baseline_lite_ap.model", help='Initial model weights');
parser.add_argument('--encoder_type', type=str,      default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args();

# ==================== INITIALISE LINE NOTIFY ====================

if ("nsml" in sys.modules):
    args.train_path     = os.path.join(DATASET_PATH[0],'train/voxceleb2')
    args.test_path      = os.path.join(DATASET_PATH[0],'train/voxceleb1')
    args.train_list     = os.path.join(DATASET_PATH[0],'train/train_list.txt')
    args.test_list      = os.path.join(DATASET_PATH[0],'train/test_list.txt')
    args.musan_path     = os.path.join(DATASET_PATH[1],'train')
    model_save_path     = "exps/model";
    result_save_path    = "exps/results"
    feat_save_path      = "feat"
else:
    model_save_path     = args.save_path+"/model"
    result_save_path    = args.save_path+"/result"
    feat_save_path      = ""

# ==================== MAKE DIRECTORIES ====================

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)
else:
    print("Folder already exists. Press Enter to continue...")

# ==================== LOAD MODEL ====================

s = SpeakerNet(**vars(args));

if("nsml" in sys.modules):
    nsml.bind(save=s.saveParameters, load=s.loadParameters);
if(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);


it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [];

trainLoader = get_data_loader(args.train_list, **vars(args))
s.make_scheduler(trainLoader, args.max_epoch, args.lr)

while(1):   
    
    trainLoader.dataset.make_spk_list()
    loss, traineer = s.train_network(loader=trainLoader);

    # ==================== EVALUATE LIST ====================

    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, feat_dir=feat_save_path, test_path=args.test_path)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TEER %2.2f, TLOSS %f, EER %2.4f"%( traineer, loss, result[1]));

        min_eer.append(result[1])
        s.current_EER = result[1]

        if ("nsml" in sys.modules):
            training_report = {};
            training_report["summary"] = True;
            training_report["epoch"] = it;
            training_report["step"] = it;
            training_report["train_loss"] = loss.item();
            training_report["val_eer"] = result[1];
            training_report["min_eer"] = min(min_eer);
            
            nsml.report(**training_report);
            nsml.save(it)
    
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TEER %2.2f, TLOSS %f"%( traineer, loss));
        

   

    # ==================== SAVE MODEL ====================

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");






