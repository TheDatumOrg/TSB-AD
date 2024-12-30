"""
This function is adapted from [NeurIPS2023-One-Fits-All] by [tianzhou2011]
Original source: [https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All]
"""

import argparse
from typing import Dict
import numpy as np
import torchinfo
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
import tqdm
import os, math
from typing import Optional
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange


from ..utils.torch_utility import EarlyStoppingTorch, PositionalEmbedding, TokenEmbedding, TemporalEmbedding, get_gpu, TimeFeatureEmbedding, DataEmbedding, adjust_learning_rate
from ..utils.dataset import ReconstructDataset    

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Model(nn.Module):
    
    def __init__(self,
                 pred_len=0,
                 seq_len=100,
                 patch_size=1,
                 stride=1,      
                 d_model = 768,
                 d_ff = 768,
                 embed = "timeF",
                 gpt_layers = 6,
                 enc_in = 1,
                 c_out = 1,
                 freq = "h",
                 dropout= 0.1,
                 mlp = 0,
                 model_path = "pre_train"):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.d_model = d_model
        self.embed = embed
        self.gpt_layers = gpt_layers
        self.enc_in = enc_in
        self.c_out = c_out
        self.freq = freq
        self.dropout = dropout
        self.model_path = model_path
        self.mlp = mlp
    
        self.patch_num = (self.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.enc_in * self.patch_size, self.d_model, self.embed, self.freq,
                                           self.dropout)

        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)    
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and self.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # if configs.use_gpu:
        #     device = torch.device('cuda:{}'.format(0))
        #     self.gpt2.to(device=device)

        self.ln_proj = nn.LayerNorm(self.d_ff)
        self.out_layer = nn.Linear(
            self.d_ff, 
            self.c_out, 
            bias=True)

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out

class OFA():
    def __init__(self,
                 win_size = 100,
                 stride = 1,
                 enc_in = 1,
                 features = 'M',
                 batch_size = 128,
                 learning_rate = 0.0001,
                 epochs = 10,
                 patience = 3,
                 lradj = "type1",
                 validation_size=0.2):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.enc_in = enc_in
        self.features = features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.lradj = lradj
        self.validation_size = validation_size

        self.decision_scores_ = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
            
        self.model = Model(seq_len=self.win_size, enc_in=self.enc_in, c_out=self.enc_in).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        self.input_shape = (self.batch_size, self.win_size, self.enc_in)
        
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size, stride=self.stride),
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
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.learning_rate)
                
            
    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size, stride=self.stride),
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
                # # criterion
                # print('batch_x: ', batch_x.shape)
                # print('outputs: ', outputs.shape)
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
        
        # Custom stride length
        scores_win = [scores[i] for i in range(scores.shape[0])]
        self.decision_scores_ = np.zeros(len(data))
        count = np.zeros(len(data))
        for i, score in enumerate(scores_win):
            start = i * self.stride
            end = start + self.win_size
            self.decision_scores_[start:end] += score
            count[start:end] += 1
        self.decision_scores_ = self.decision_scores_ / np.maximum(count, 1)

        return self.decision_scores_
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
