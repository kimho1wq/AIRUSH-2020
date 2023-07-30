#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV
from loss.angleproto import AngleProtoLoss

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + numpy.cos(step / total_steps * numpy.pi))

class SpeakerNet(nn.Module):

    def __init__(self, max_frames, lr = 0.0001, model="ResNetSE34", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='angleproto', **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type}

        SpeakerNetModel = importlib.import_module('models.%s'%(model)).__getattribute__(model)
        self.__S__ = SpeakerNetModel(**argsdict).cuda();

        if trainfunc == 'angleproto':
            self.__L__ = AngleProtoLoss().cuda()
            self.__train_normalize__    = True
            self.__test_normalize__     = True
        else:
            raise ValueError('Undefined loss.')

        if optimizer == 'adam':
            self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr);
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9, weight_decay=5e-5);
        else:
            raise ValueError('Undefined optimizer.')
        
        self.__max_frames__ = max_frames;
        self.current_EER = 50.0

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train contrastive
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, print_interval=100):
        clr = []
        for param_group in self.__optimizer__.param_groups:
            clr.append(param_group['lr'])
       
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Training with LR %.5f..."%(max(clr)))
        self.train();

        # ==================== INITIAL PARAMETERS ====================

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0    
        
        for data in loader:

            tstart = time.time()

            self.zero_grad();
            
            org_size = data.size()
            data = data.reshape(org_size[0] * org_size[1], -1)
            outp      = self.__S__.forward(data.cuda())
            if self.__train_normalize__:
                outp   = F.normalize(outp, p=2, dim=1)
            feat = outp.reshape(org_size[0], org_size[1], -1)

            nloss, prec1 = self.__L__.forward(feat)

            loss    += nloss.detach().cpu();
            top1    += prec1
            counter += 1;
            index   += stepsize;

            nloss.backward();
            self.__optimizer__.step();
            self.scheduler.step()

            telapsed = time.time() - tstart

            if counter % print_interval == 1:
                print("Processing (%d) Loss %f ACC/T1 %2.3f%% - %.2f Hz "%(index, loss/counter, top1/counter, stepsize/telapsed))
                
        return (loss/counter, top1/counter);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Read data from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def readDataFromList(self, listfilename):

        data_list = {};

        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if not line:
                    break;

                data = line.split();
                filename = data[1];
                speaker_name = data[0]

                if not (speaker_name in data_list):
                    data_list[speaker_name] = [];
                data_list[speaker_name].append(filename);

        return data_list


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromListSave(self, listfilename, print_interval=5000, feat_dir='', test_path='', num_eval=10):
        
        self.eval();
        
        lines       = []
        files       = []
        filedict    = {}
        feats       = {}
        tstart      = time.time()

        if feat_dir != '':
            print('Saving temporary files to %s'%feat_dir)
            if not(os.path.exists(feat_dir)):
                os.makedirs(feat_dir)

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): #  or (len(all_scores)==1000) 
                    break;

                data = line.split();

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), self.__max_frames__, evalmode=True, num_eval=num_eval)).cuda()
            
            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            if feat_dir == '':
                feats[file]     = ref_feat
            else:
                filedict[file]  = filename
                torch.save(ref_feat,os.path.join(feat_dir,filename))

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                print("Reading %d: %.2f Hz, embed size %d"%(idx,idx/telapsed,ref_feat.size()[1]))
                
        print('')
        all_scores = [];
        all_labels = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            if feat_dir == '':
                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()
            else:
                ref_feat = torch.load(os.path.join(feat_dir,filedict[data[1]])).cuda()
                com_feat = torch.load(os.path.join(feat_dir,filedict[data[2]])).cuda()

            if self.__test_normalize__:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1).expand(-1,-1,num_eval), com_feat.unsqueeze(-1).expand(-1,-1,num_eval).transpose(0,2)).detach().cpu().numpy();

            score = -1 * numpy.mean(dist);

            all_scores.append(score);  
            all_labels.append(int(data[0]));

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                print("Computing %d: %.2f Hz"%(idx,idx/telapsed))
                
        if feat_dir != '':
            print(' Deleting temporary files.')
            shutil.rmtree(feat_dir)

        print('\n')

        return (all_scores, all_labels);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def make_scheduler(self, loader, max_epoch, lr, min_lr = 0.000005):
        total_steps = max_epoch * len(loader)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.__optimizer__,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  
                min_lr / lr))


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.state_dict(), path + '/pretrained.pt');


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParametersModelEnsemble(self, paths):

        self_state = self.state_dict();

        states = []
        for path in paths:
            states.append(torch.load(path));

        loaded_state = states[0]

        for state in states[1:]:
            for name, param in state.items():
                loaded_state[name] = loaded_state[name] + param

        for name, param in state.items():
            loaded_state[name] = loaded_state[name] / len(paths)

        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);