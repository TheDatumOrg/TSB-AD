"""
This function is adapted from [OmniAnomaly] by [TsingHuasuya et al.]
Original source: [https://github.com/NetManAIOps/OmniAnomaly]
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu

class OmniAnomalyModel(nn.Module):
    def __init__(self, feats, device):
        super(OmniAnomalyModel, self).__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            # nn.Flatten(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden = None):
        bs = x.shape[0]
        win = x.shape[1]

        # hidden = torch.rand(2, bs, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        hidden = torch.rand(2, bs, self.n_hidden).to(self.device) if hidden is not None else hidden

        out, hidden = self.lstm(x.view(-1, bs, self.n_feats), hidden)

        # print('out: ', out.shape)       # (L, bs, n_hidden)
        # print('hidden: ', hidden.shape) # (2, bs, n_hidden)

        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        ## Decoder
        x = self.decoder(x)             # (L, bs, n_feats)
        return x.reshape(bs, win*self.n_feats), mu.reshape(bs, win*self.n_latent), logvar.reshape(bs, win*self.n_latent), hidden


class OmniAnomaly(BaseDetector):
    def __init__(self,
                 win_size = 5,
                 feats = 1,
                 batch_size = 128,
                 epochs = 50,
                 patience = 3,
                 lr = 0.002,
                 validation_size=0.2
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size

        self.model = OmniAnomalyModel(feats=self.feats, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss(reduction = 'none')

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

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
        
        mses, klds = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:        
                d = d.to(self.device)
                # print('d: ', d.shape)

                y_pred, mu, logvar, hidden = self.model(d, hidden if idx else None)
                d = d.view(-1, self.feats*self.win_size)
                MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                loss = torch.mean(MSE + self.model.beta * KLD)

                mses.append(torch.mean(MSE).item())
                klds.append(self.model.beta * torch.mean(KLD).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (d, _) in loop:
                        d = d.to(self.device)
                        y_pred, mu, logvar, hidden = self.model(d, hidden if idx else None)
                        d = d.view(-1, self.feats*self.win_size)
                        MSE = torch.mean(self.criterion(y_pred, d), axis=-1)
                        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                        loss = torch.mean(MSE + self.model.beta * KLD)

                        avg_loss_val += loss.cpu().item()
                        loop.set_description(
                            f"Validation Epoch [{epoch}/{self.epochs}]"
                        )
                        loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

            self.scheduler.step()
            if len(valid_loader) > 0:
                avg_loss = avg_loss_val / len(valid_loader)
            else:
                avg_loss = avg_loss / len(train_loader)
            self.early_stopping(avg_loss, self.model)
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
        y_preds = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (d, _) in loop:
                d = d.to(self.device)
                # print('d: ', d.shape)

                y_pred, _, _, hidden = self.model(d, hidden if idx else None)
                y_preds.append(y_pred)
                d = d.view(-1, self.feats*self.win_size)

                # print('y_pred: ', y_pred.shape)
                # print('d: ', d.shape)
                loss = torch.mean(self.criterion(y_pred, d), axis=-1)
                # print('loss: ', loss.shape)

                scores.append(loss.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
