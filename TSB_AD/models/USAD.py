"""
This function is adapted from [usad] by [manigalati]
Original source: [https://github.com/manigalati/usad]
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

class USADModel(nn.Module):
    def __init__(self, feats, n_window=5):
        super(USADModel, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = n_window # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        bs = g.shape[0]
        ## Encode
        # z = self.encoder(g.view(1,-1))
        z = self.encoder(g.view(bs, self.n))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        # return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)
        return ae1.view(bs, self.n), ae2.view(bs, self.n), ae2ae1.view(bs, self.n)


class USAD(BaseDetector):
    def __init__(self,
                 win_size = 5,
                 feats = 1,
                 batch_size = 128,
                 epochs = 10,
                 patience = 3,
                 lr = 1e-4,
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

        self.model = USADModel(feats=self.feats, n_window=self.win_size).to(self.device)
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
        
        l1s, l2s = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:        
                d = d.to(self.device)     # (bs, win, feat)
                # print('d: ', d.shape)

                ae1s, ae2s, ae2ae1s = self.model(d)
                # print('ae2ae1s: ', ae2ae1s.shape)

                d = d.view(ae2ae1s.shape[0], self.feats*self.win_size)

                l1 = (1 / n) * self.criterion(ae1s, d) + (1 - 1/n) * self.criterion(ae2ae1s, d)
                l2 = (1 / n) * self.criterion(ae2s, d) - (1 - 1/n) * self.criterion(ae2ae1s, d)
                # print('l1: ', l1.shape)

                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)

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
                        ae1s, ae2s, ae2ae1s = self.model(d)
                        d = d.view(ae2ae1s.shape[0], self.feats*self.win_size)

                        l1 = (1 / n) * self.criterion(ae1s, d) + (1 - 1/n) * self.criterion(ae2ae1s, d)
                        l2 = (1 / n) * self.criterion(ae2s, d) - (1 - 1/n) * self.criterion(ae2ae1s, d)

                        l1s.append(torch.mean(l1).item())
                        l2s.append(torch.mean(l2).item())
                        loss = torch.mean(l1 + l2)
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
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for idx, (d, _) in loop:
                d = d.to(self.device)
                # print('d: ', d.shape)

                ae1, ae2, ae2ae1 = self.model(d)
                d = d.view(ae2ae1.shape[0], self.feats*self.win_size)

                # print('ae2ae1: ', ae2ae1.shape)
                # print('d: ', d.shape)

                loss = 0.1 * self.criterion(ae1, d) + 0.9 * self.criterion(ae2ae1, d)
                # print('loss: ', loss.shape)
                loss = torch.mean(loss, axis=-1)

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
