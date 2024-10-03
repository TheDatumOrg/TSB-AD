"""
This function is adapted from [FITS] by [VEWOXIC]
Original source: [https://github.com/VEWOXIC/FITS]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import torchinfo
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import math

from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ReconstructDataset    

class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, seq_len, pred_len, individual, enc_in, cut_freq):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in

        self.dominance_freq = cut_freq # 720/24
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len


    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # energy compemsation for the length change
        # dom_x=x-low_x
        
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy, low_xy* torch.sqrt(x_var)

    
class FITS():
    def __init__(self,
                 win_size=100,
                 DSR=4,
                 individual=True,
                 input_c=1,
                 batch_size=128,
                 cut_freq=12,
                 epochs=50,
                 lr=1e-3,
                 validation_size=0.2
                 ):
        super().__init__()
        self.__anomaly_score = None
        
        self.cuda = True
        self.device = get_gpu(self.cuda)

            
        self.win_size = win_size        
        self.DSR = DSR
        self.individual = individual
        self.input_c = input_c
        self.batch_size = batch_size
        self.cut_freq = cut_freq
        self.validation_size = validation_size

        self.model = Model(seq_len=self.win_size//self.DSR, pred_len=self.win_size-self.win_size//self.DSR, individual=self.individual, enc_in=self.input_c, cut_freq=self.cut_freq).to(self.device)

        self.epochs = epochs
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
    
    def fit(self, data):
        
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for idx, (x, target) in loop:

                x = x[:, ::self.DSR, :]
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                output, _ = self.model(x)

                # print('x: ', x.shape)
                # print('target: ', target.shape)
                
                loss = self.loss(output, target)
                loss.backward()

                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            
            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:

                    x = x[:, ::self.DSR, :]
                    x, target = x.to(self.device), target.to(self.device)
                    output, _ = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
    
    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:

                x = x[:, ::self.DSR, :]
                x, target = x.to(self.device), target.to(self.device)
                output, _ = self.model(x)
                # loss = self.loss(output, target)
                score = torch.mean(self.anomaly_criterion(output, target), dim=-1)
                scores.append(score.cpu()[:,-1])

                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy().flatten()

        assert scores.ndim == 1
        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score


    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.input_len), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))