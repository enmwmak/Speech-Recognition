#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mwmak
Train a CNN model for spoken-digit recognition
Example usage:
    python3 digitrec.py --pool_method stats --model_file models/spokendigit_cnn_stats.pth
    python3 digitrec.py --pool_method adapt --model_file models/spokendigit_cnn_adapt.pth
    python3 digitrec.py --pool_method none --model_file models/spokendigit_cnn_none.pth
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import numpy as np
import torch
from tqdm import tqdm
from model import CNNModel
from pathlib import Path
import argparse

class SpeechDataset(Dataset):
    def __init__(self, filelist = None, rootdir = None, n_mfcc=20, max_len=16000):
        self.filelist = filelist
        self.df = pd.read_csv(filelist, names=['file'])
        self.rootdir = rootdir
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.labels = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                       'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.loc[i]
        label = self.labels[row.str.split('/')[0][0]]
        waveform, srate = librosa.load(self.rootdir + '/' + row['file'])
        cur_len = waveform.shape[0]
        if cur_len > self.max_len:
            out_wav = waveform[0:self.max_len]      # Trim the tail
        else:
            out_wav = np.zeros((self.max_len,))     # Pad tail with 0     
            out_wav[0:cur_len] = waveform
        mfcc = librosa.feature.mfcc(y=out_wav, sr=srate, n_mfcc=self.n_mfcc,
                                    n_fft=512, hop_length=160, center=False)
        mfcc = torch.from_numpy(mfcc.astype(float))
        label = np.array(label, dtype=np.int64)
        label = torch.tensor(label)
        return mfcc, label


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
def evaluate(model, loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in loader]
    outputs = torch.tensor(outputs).T
    loss, accuracy = torch.mean(outputs, dim=1)
    return {"loss" : loss.item(), "accuracy" : accuracy.item()}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(model, train_loader, val_loader, epochs, lr, optimizer_function=torch.optim.Adam):
    history = []
    optimizer = optimizer_function(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        print("Epoch ", epoch)

        #Train
        model.train()
        lrs = []
        tr_loss = []
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss = model.training_step(batch)
            tr_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()

        #Validate
        result = evaluate(model, val_loader)
        result["lrs"] = lrs
        result["train loss"] = torch.stack(tr_loss).mean().item()
        acc = result['accuracy']*100
        print("Last lr: ", lrs[-1]," Train_loss: ", result["train loss"], " Val_loss: ", 
              result['loss'], " Accuracy: %.2f%%" % acc)
        history.append(result)         
        
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_method', choices=['stats', 'adapt', 'none'], required=True)
    parser.add_argument('--model_file', required=True)
    args = parser.parse_args()

    train_set = SpeechDataset(filelist='data/digits/short_train.lst', rootdir='data/digits', n_mfcc=20)
    test_set = SpeechDataset(filelist='data/digits/short_test.lst', rootdir='data/digits', n_mfcc=20)
    train_dl = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=16, pin_memory=True) 
    test_dl = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16, pin_memory=True) 
    
    device = get_default_device()
    model = CNNModel(pool_method=args.pool_method).to(device)
    fit(model, train_dl, test_dl, epochs=10, lr=0.001)
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_file)

    """
    device = get_default_device()
    model = ResNetModel(pool_method=args.pool_method).to(device)
    fit(model, train_dl, test_dl, epochs=10, lr=0.001)
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_file)
    """
