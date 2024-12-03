'''
TimesNet from "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
Code partially from https://github.com/thuml/Time-Series-Library/

Copyright (c) 2021 THUML @ Tsinghua University
'''

from typing import Dict
import numpy as np
import torchinfo
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.fft
from torch.nn.utils import weight_norm
import math
import tqdm
import os

from ..utils.torch_utility import EarlyStoppingTorch, DataEmbedding, adjust_learning_rate, get_gpu
from ..utils.dataset import ReconstructDataset    
 
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self,
                 seq_len=96,
                 pred_len=0,
                 top_k=3,
                 d_model=8,
                 d_ff=16,
                 num_kernels=6
                 ):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self,
                 seq_len=96,
                 pred_len=0,
                 d_model=8,
                 enc_in=1,
                 c_out=1,
                 e_layers=1,
                 dropout=0.1,
                 embed='timeF',
                 freq="t"
                 ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len=self.seq_len)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)


    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]

class TimesNet():
    def __init__(self,
                 win_size=96,
                 enc_in=1,
                 epochs=10,
                 batch_size=128,
                 lr=1e-4,
                 patience=3,
                 features="M",
                 lradj="type1",
                 validation_size=0.2):
        super().__init__()

        self.win_size = win_size
        self.enc_in = enc_in
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.features = features
        self.lradj = lradj
        self.validation_size = validation_size

        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
            
        self.model = Model(seq_len=self.win_size, enc_in=self.enc_in, c_out=self.enc_in).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        
        self.input_shape = (self.batch_size, self.win_size, self.enc_in)
        
    
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
        
        train_steps = len(train_loader)
        for epoch in range(1, self.epochs + 1):
            ## Training
            train_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, (batch_x, _) in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
            
            ## Validation
            self.model.eval()
            total_loss = []
            
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, (batch_x, _) in loop:
                    batch_x = batch_x.float().to(self.device)

                    outputs = self.model(batch_x)

                    f_dim = -1 if self.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()

                    loss = self.criterion(pred, true)
                    total_loss.append(loss)
                    loop.set_description(f'Valid Epoch [{epoch}/{self.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.lr)
                        
    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        attens_energy = []
        y_hats = []
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        
        loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
        with torch.no_grad():
            for i, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                y_hat = torch.squeeze(outputs, -1)
                
                score = score.detach().cpu().numpy()[:, -1]
                y_hat = y_hat.detach().cpu().numpy()[:, -1]
                
                attens_energy.append(score)
                y_hats.append(y_hat)
                loop.set_description(f'Testing Phase: ')

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        scores = np.array(attens_energy)
        
        y_hats = np.concatenate(y_hats, axis=0).reshape(-1)
        y_hats = np.array(y_hats)

        assert scores.ndim == 1
        
        import shutil
        self.save_path = None
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores
        self.y_hats = y_hats

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))