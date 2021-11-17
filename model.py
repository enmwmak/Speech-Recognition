#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define network models
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class StatsPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)


class CNNModel(nn.Module):
    def __init__(self, n_mfcc=20, n_classes=10, pool_method='None'):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_classes =  n_classes
        self.conv1 = nn.Sequential(nn.Conv1d(n_mfcc, 16, kernel_size=3, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))
        if pool_method == 'adapt':
            self.pool = nn.AdaptiveAvgPool2d((128,1))
            self.emb_size = 128            
        elif pool_method == 'stats':    
            self.pool = StatsPooling()
            self.emb_size = 256
        else:
            self.pool = nn.Identity()
            self.emb_size = 6*128              # For 1-sec of speech with hop_length=160, 16kHz
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(self.emb_size, 64), nn.ReLU(), 
                                 nn.Linear(64, self.n_classes), nn.Sigmoid())
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        out = self.fc(x)
        return(out)

    def training_step(self, batch):
        inputs, labels = batch
        inputs = inputs.float()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def validation_step(self, batch):
        inputs, labels = batch
        inputs = inputs.float()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, pred = torch.max(outputs, 1)
        accuracy = torch.tensor(torch.sum(pred==labels).item()/len(pred))
        return [loss.detach(), accuracy.detach()] 


