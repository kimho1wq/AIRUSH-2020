import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
from math import ceil

from sklearn.metrics import f1_score
import nsml

def bind_one_model(model):

        def save(model_dir):
            checkpoint = {
                'model': model.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_dir, 'model'))

        def load(model_dir, **kwargs):
            fpath = os.path.join(model_dir, 'model')
            checkpoint = torch.load(fpath)
            model.load_state_dict(checkpoint['model'])

        def infer(test_dir, **kwargs):
            return 

        nsml.bind(save=save, load=load, infer=infer)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))

def make_scheduler(optimizer, loader, max_epoch, lr, min_lr = 0.000005):
        total_steps = max_epoch * len(loader)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  
                min_lr / lr))

        return scheduler

class MusicDataset(Dataset):
    def __init__(self, config, dataset_root, qn, train=True):
        self.dataset_name = config['dataset_name_%s'%(qn)]  # i.e. q1
        self.input_length = config['input_length']
        self.train = train
        self.mel_dir = os.path.join(dataset_root, 'train_data', 'mel_spectrogram')
        self.label_file = os.path.join(dataset_root, 'train_label')

        with open(self.label_file) as f:
            self.train_labels = json.load(f)

        self.n_valid = len(self.train_labels['track_index']) // 1000
        self.n_train = len(self.train_labels['track_index']) - self.n_valid

        label_types = {'q1': 'station_name',
                       'q2': 'mood_tag',
                       'q3': 'genre'}

        self.label_type = label_types[self.dataset_name]
        self.label_map = self.create_label_map()
        self.n_classes = len(self.label_map)
    def create_label_map(self):
        label_map = {}
        if self.dataset_name in ['q1', 'q3']:
            for idx, label in self.train_labels[self.label_type].items():
                if label not in label_map:
                    label_map[label] = len(label_map)
        else:
            for idx, label_list in self.train_labels[self.label_type].items():
                for label in label_list:
                    if label not in label_map:
                        label_map[label] = len(label_map)

        return label_map

    def __getitem__(self, idx):
        data_idx = str(idx)
        if not self.train:
            data_idx = str(idx + self.n_train)

        track_name = self.train_labels['track_index'][data_idx]
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name))
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10 * mel)
        mel_time = mel.shape[-1]

        if mel_time > self.input_length:
            start_t = np.random.randint(low=0, high=mel_time-self.input_length)
            mel = mel[:, start_t : start_t+self.input_length]
        elif mel_time < self.input_length:
            nb_dup = int(self.input_length/ mel_time)+1
            mel = np.tile(mel, (1, nb_dup))[:, :self.input_length]
        else:
            mel = mel

        label = self.train_labels[self.label_type][data_idx]
        if self.dataset_name in ['q1', 'q3']:
            labels = self.label_map[label]
        else:
            label_idx = [self.label_map[l] for l in label]
            labels = np.zeros(self.n_classes, dtype=np.float32)
            labels[label_idx] = 1


        return mel, labels

    def __len__(self):
        return self.n_train if self.train else self.n_valid


class TestMusicDataset(Dataset):
    def __init__(self, config, qn, dataset_root):
        self.dataset_name = config['dataset_name_%s'%(qn)]  # i.e. q1
        self.meta_dir = os.path.join(dataset_root, 'test_data', 'meta')
        self.mel_dir = os.path.join(dataset_root, 'test_data', 'mel_spectrogram')

        meta_path = os.path.join(self.meta_dir, '{}_test.json'.format(self.dataset_name))
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.input_length = config['input_length']
        self.n_classes = 100 if self.dataset_name == 'q2' else 4

    def __getitem__(self, idx):
        data_idx = str(idx)

        track_name = self.meta['track_index'][data_idx]
        mel_path = os.path.join(self.mel_dir, '{}.npy'.format(track_name))
        mel = np.load(mel_path)[0]
        mel = np.log10(1 + 10 * mel)

        mel_time = mel.shape[-1]
        if mel_time > self.input_length:
            start_t = np.random.randint(low=0, high=mel_time-self.input_length)
            mel = mel[:, start_t : start_t+self.input_length]
        elif mel_time < self.input_length:
            nb_dup = int(self.input_length/ mel_time)+1
            mel = np.tile(mel, (1, nb_dup))[:, :self.input_length]
        else:
            mel = mel
            

        return mel, data_idx

    def __len__(self):
        return len(self.meta['track_index'])


def accuracy_(pred, target):
    _, predicted_max_idx = pred.max(dim=1)
    n_correct = predicted_max_idx.eq(target).sum().item()
    return n_correct / len(target)


def f1_score_(pred, target, threshold=0.5):
    pred = np.array(pred.cpu() > threshold, dtype=float)
    return f1_score(target.cpu(), pred, average='micro')


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)



class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpeakerNet(nn.Module):
    def __init__(self, block, layers, num_filters, encoder_type="SAP"):
        
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(SpeakerNet, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d((2,2))

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1])
        self.layer3 = self._make_layer(block, num_filters[2], layers[2])
        self.layer4 = self._make_layer(block, num_filters[3], layers[3])
        self.maxpool1 = nn.MaxPool2d((2,2))
        self.maxpool2 = nn.MaxPool2d((2,4))
        self.maxpool3 = nn.MaxPool2d((4,4))
        self.maxpool4 = nn.MaxPool2d((4,4))


        self.bn_before_gru = nn.BatchNorm2d(num_features = num_filters[3])
        self.gru = nn.GRU(input_size = num_filters[3],
			hidden_size = 128,
			num_layers = 1,
			batch_first = True)


        self.fc1_1 = nn.Linear(128, 256)
        self.fc1_2 = nn.Linear(256, 100)
  
        self.fc2_1 = nn.Linear(128, 128)
        self.fc2_2 = nn.Linear(128, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, qn):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool0(x)

        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.maxpool3(x)
        x = self.layer4(x)
        x = self.maxpool4(x)

        
        x = self.bn_before_gru(x)
        x = self.relu(x)
        x = x.squeeze(dim=2).permute(0, 2, 1)  #(batch, channel, time) >> (batch, time, channel)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]

        if qn == '2':
            x = self.fc1_1(x)
            x = self.fc1_2(x)
            return x
        if qn == '1':
            x = self.fc2_1(x)
            x = self.fc2_2(x)
            return x

        return x


class SEBasicBlock_1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(SEBasicBlock_1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer_1d(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out



class SELayer_1d(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SpeakerNet_1d(nn.Module):
    def __init__(self, block, layers, num_filters, encoder_type="SAP"):
        
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(SpeakerNet_1d, self).__init__()

        self.conv1 = nn.Conv1d(128, num_filters[0] , kernel_size=7, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool1d(2)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1])
        self.layer3 = self._make_layer(block, num_filters[2], layers[2])
        self.layer4 = self._make_layer(block, num_filters[3], layers[3])
        self.maxpool1 = nn.MaxPool1d(2)
        self.maxpool2 = nn.MaxPool1d(4)
        self.maxpool3 = nn.MaxPool1d(4)
        self.maxpool4 = nn.MaxPool1d(4)


        self.bn_before_gru = nn.BatchNorm1d(num_features = num_filters[3])
        self.gru = nn.GRU(input_size = num_filters[3],
			hidden_size = 128,
			num_layers = 1,
			batch_first = True)


        self.fc1_1 = nn.Linear(128, 256)
        self.fc1_2 = nn.Linear(256, 100)
  
        self.fc2_1 = nn.Linear(128, 128)
        self.fc2_2 = nn.Linear(128, 4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, qn):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool0(x)

        x = self.layer1(x)
        x = self.maxpool1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)
        x = self.layer3(x)
        x = self.maxpool3(x)
        x = self.layer4(x)
        x = self.maxpool4(x)

        
        x = self.bn_before_gru(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  #(batch, channel, time) >> (batch, time, channel)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]

        if qn == '2':
            x = self.fc1_1(x)
            x = self.fc1_2(x)
            return x
        if qn == '1':
            x = self.fc2_1(x)
            x = self.fc2_2(x)
            return x

        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_swish(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(Swish())


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            _add_conv_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        _add_conv(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels,
                  active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        return out

    
class ResNeXt50(nn.Module):
    def __init__(self):
        super(ResNeXt50, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = models.resnext50_32x4d().bn1
        self.relu = models.resnext50_32x4d().relu
        self.maxpool = models.resnext50_32x4d().maxpool
        self.layer1 = models.resnext50_32x4d().layer1
        self.layer2 = models.resnext50_32x4d().layer2
        self.layer3 = models.resnext50_32x4d().layer3
        self.layer4 = models.resnext50_32x4d().layer4
        self.avgpool = models.resnext50_32x4d().avgpool
        self.fc = nn.Sequential(
            nn.Linear(2048, 128, bias=True),          
            nn.Linear(128, 4, bias=True))


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class RexNet(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=100,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(RexNet, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [2, 2, 2, 2, 2, 2]
        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        self.depth = sum(layers[:]) * 3

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        _add_conv_swish(features, 1, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        if use_se:
            use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
        else:
            use_ses = [False] * sum(layers[:])

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        _add_conv_swish(features, c, pen_channels)
        self.features = nn.Sequential(*features)

        self.bn_before_gru = nn.BatchNorm2d(num_features = pen_channels)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size = pen_channels,
			hidden_size = 512,
			num_layers = 1,
			batch_first = True)

        self.fc1_1 = nn.Linear(512, 256)
        self.fc1_2 = nn.Linear(256, 100)
  
        self.fc2_1 = nn.Linear(512, 128)
        self.fc2_2 = nn.Linear(128, 4)

    def forward(self, x, qn):
        x = torch.unsqueeze(x, dim=1)
        x = self.features(x)

        x = self.bn_before_gru(x)
        x = self.relu(x)
        x = x.squeeze(dim=2).permute(0, 2, 1)  #(batch, channel, time) >> (batch, time, channel)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]

        if qn == '2':
            x = self.fc1_1(x)
            x = self.fc1_2(x)
            return x
        if qn == '1':
            x = self.fc2_1(x)
            x = self.fc2_2(x)
            return x

        return x



class Trainer:
    def __init__(self, config, mode):
        """
        mode: train(run), test(submit)
        """
        self.device = config['device']
        self.dataset_name = {}
        self.dataset_name['1'] = config['dataset_name_1']
        self.dataset_name['2'] = config['dataset_name_2']
        self.config = config

        if mode == 'train':
            batch_size = config['batch_size']
            self.train_dataset = {}
            self.valid_dataset = {}
            self.label_map = {}
            self.train_loader = {}
            self.valid_loader = {}

            self.train_dataset['1'] = MusicDataset(config, config['dataset_root_1'], '1', train=True)
            self.valid_dataset['1'] = MusicDataset(config, config['dataset_root_1'], '1', train=False)
            self.train_dataset['2'] = MusicDataset(config, config['dataset_root_2'], '2', train=True)
            self.valid_dataset['2'] = MusicDataset(config, config['dataset_root_2'], '2', train=False)
            self.label_map['1'] = self.train_dataset['1'].label_map
            self.label_map['2'] = self.train_dataset['2'].label_map

            self.train_loader['1'] = DataLoader(self.train_dataset['1'], batch_size=batch_size, shuffle=True)
            self.valid_loader['1'] = DataLoader(self.valid_dataset['1'], batch_size=batch_size, shuffle=False)
            self.train_loader['2'] = DataLoader(self.train_dataset['2'], batch_size=batch_size, shuffle=True)
            self.valid_loader['2'] = DataLoader(self.valid_dataset['2'], batch_size=batch_size, shuffle=False)

        self.criterion = {}
        self.act = {}
        self.act_kwargs = {}
        self.measure_name = {}
        self.measure_fn = {}

        if self.dataset_name['1'] in ['q1', 'q3']:
            config['n_classes1'] = 4
            self.criterion['1'] = nn.CrossEntropyLoss()
            self.act['1'] = torch.nn.functional.log_softmax
            self.act_kwargs['1'] = {'dim': 1}
            self.measure_name['1'] = 'accuracy'
            self.measure_fn['1'] = accuracy_
        else:
            config['n_classes1'] = 100
            self.criterion['1'] = nn.BCEWithLogitsLoss()
            self.act['1'] = torch.sigmoid
            self.act_kwargs['1'] = {}
            self.measure_name['1'] = 'f1_score'
            self.measure_fn['1'] = f1_score_

        if self.dataset_name['2'] in ['q1', 'q3']:
            config['n_classes2'] = 4
            self.criterion['2'] = nn.CrossEntropyLoss()
            self.act['2'] = torch.nn.functional.log_softmax
            self.act_kwargs['2'] = {'dim': 1}
            self.measure_name['2'] = 'accuracy'
            self.measure_fn['2'] = accuracy_
        else:
            config['n_classes1'] = 100
            self.criterion['2'] = nn.BCEWithLogitsLoss()
            self.act['2'] = torch.sigmoid
            self.act_kwargs['2'] = {}
            self.measure_name['2'] = 'f1_score'
            self.measure_fn['2'] = f1_score_


        num_filters = [16, 32, 64, 128]
        layers = [3, 4, 6, 3]
        self.model0 = SpeakerNet(SEBasicBlock, layers, num_filters).to(self.device)
        bind_one_model(self.model0)
        nsml.load(checkpoint='60_1', session='t0023/rush4-2/264')

        self.model1 = SpeakerNet_1d(SEBasicBlock_1d, layers, num_filters).to(self.device)
        bind_one_model(self.model1)
        nsml.load(checkpoint='330', session='t0023/rush4-3/54')

        self.model2 = ResNeXt50().to(self.device)
        bind_one_model(self.model2)
        nsml.load(checkpoint='430', session='t0023/rush4-2/138')

        self.model3 = RexNet().to(self.device)
        bind_one_model(self.model3)
        nsml.load(checkpoint='260', session='t0023/rush4-3/136')


        nb_params = sum([param.view(-1).size()[0] for param in self.model0.parameters()])
        nb_params = sum([param.view(-1).size()[0] for param in self.model1.parameters()])
        nb_params = sum([param.view(-1).size()[0] for param in self.model2.parameters()])
        nb_params = sum([param.view(-1).size()[0] for param in self.model3.parameters()])


        if mode == 'train':
            self.scheduler = {}
            self.optimizer = torch.optim.Adam(self.model0.parameters(), lr = 0.001, weight_decay = 1e-4)
            self.scheduler['1'] = make_scheduler(self.optimizer, self.train_loader['1'], max_epoch = 500, lr = 0.001)
            self.scheduler['2'] = make_scheduler(self.optimizer, self.train_loader['2'], max_epoch = 500, lr = 0.001)

        self.iter = config['iter']
        self.val_iter = config['val_iter']
        self.save_iter = config['save_iter']


    def run_evaluation(self, qn, test_dir):
        """
        Predicted Labels should be a list of labels / label_lists
        """
        print('self.dataset_name[qn]',self.dataset_name[qn])
        dataset = TestMusicDataset(self.config, qn, test_dir)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.model0.eval()
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        idx2label = {v: k for k, v in self.label_map[qn].items()}

        predicted_labels = []
        for x, data_idx in loader:
            x = x.to(self.device)
            y0_ = self.model0(x, qn)
            y1_ = self.model1(x, qn)
            y2_ = self.model2(x)
            #y3_ = self.model3(x, qn)

            if self.dataset_name[qn] in ['q1', 'q3']:
                y0_ = F.softmax(y0_, dim=1)
                y1_ = F.softmax(y1_, dim=1)
                y2_ = F.softmax(y2_, dim=1)
                y_ = y0_*0.365+ y1_*0.32+ y2_*0.3150 #score= 0.965

                predicted_probs, predicted_max_idx = y_.max(dim=1)
                predicted_labels += list(predicted_max_idx)
            else:
                threshold = 0.5
                y_ = self.act[qn](y_.detach(), **self.act_kwargs[qn])
                over_threshold = np.array(y_.cpu() > threshold, dtype=float)
                label_idx_list = [np.where(labels == 1)[0].tolist() for labels in over_threshold]
                predicted_labels += label_idx_list

        if self.dataset_name[qn] in ['q1', 'q3']:
            predicted_labels = [idx2label[label_idx.item()] for label_idx in predicted_labels]
        else:
            predicted_labels = [[idx2label[label_idx] for label_idx in label_idx_list] for label_idx_list in
                                predicted_labels]

        return predicted_labels

    def run(self):
        self.save('0')
        exit()
   

    def save(self, epoch):
        nsml.save(epoch)

    def report(self, epoch, status):
        print(status)
        nsml.report(summary=True, scope=locals(), step=epoch, **status)


