"""
This function is adapted from [M2N2] by [Dongmin Kim et al.]
Original source: [https://github.com/carrtesy/M2N2]
Reimplemented by: [EmorZz1G]
"""


import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import math

# Trainer

class Trainer:
    '''
    Prepares Offline-trained models.
    '''
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader


    def train(self):
        raise NotImplementedError()

    @DeprecationWarning
    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)

    @DeprecationWarning
    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)


    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)
            
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle
import os

import pandas as pd

import json
from ast import literal_eval



# matplotlib.rcParams['agg.path.chunksize'] = 10000


class Tester:
    '''
    Test-time logics,
    including offline evaluation and online adaptation.
    '''
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader


    def calculate_anomaly_scores(self, dataloader):
        raise NotImplementedError()


    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)


    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)


    def load_trained_model(self):
        self.load(os.path.join(self.args.checkpoint_path, f"best.pth"))


    @staticmethod
    def save_dictionary(dictionary, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(dictionary, f)


    def prepare_stats(self):
        '''
        prepare anomaly scores of train data / test data.
        '''
        # train
        train_anoscs_pt_path = os.path.join(self.args.output_path, "train_anoscs.pt")
        if self.args.load_anoscs and os.path.isfile(train_anoscs_pt_path):
            self.logger.info("train_anoscs.pt file exists, loading...")
            with open(train_anoscs_pt_path, 'rb') as f:
                train_anoscs = torch.load(f)
                train_anoscs.to(self.args.device)
            self.logger.info(f"{train_anoscs.shape}")
        else:
            self.logger.info("train_anoscs.pt file does not exist, calculating...")
            train_anoscs = torch.Tensor(self.calculate_anomaly_scores(self.train_loader))  # (B, L, C) => (B, L)
            self.logger.info("saving train_anoscs.pt...")
            with open(train_anoscs_pt_path, 'wb') as f:
                torch.save(train_anoscs, f)
        torch.cuda.empty_cache()

        # test
        test_anosc_pt_path = os.path.join(self.args.output_path, "test_anoscs.pt")
        if self.args.load_anoscs and os.path.isfile(test_anosc_pt_path):
            self.logger.info("test_anoscs.pt file exists, loading...")
            with open(test_anosc_pt_path, 'rb') as f:
                test_anoscs = torch.load(f)
                test_anoscs.to(self.args.device)
            self.logger.info(f"{test_anoscs.shape}")
        else:
            self.logger.info("test_anoscs.pt file does not exist, calculating...")
            test_anoscs = torch.Tensor(self.calculate_anomaly_scores(self.test_loader))  # (B, L, C) => (B, L)
            self.logger.info("saving test_anoscs.pt...")
            with open(test_anosc_pt_path, 'wb') as f:
                torch.save(test_anoscs, f)
        torch.cuda.empty_cache()

        # train_anoscs, test anoscs (T=B*L, ) and ground truth
        train_mask = (self.train_loader.dataset.y != -1)
        self.train_anoscs = train_anoscs.detach().cpu().numpy()[train_mask] # does not include -1's
        self.test_anoscs = test_anoscs.detach().cpu().numpy() # may include -1's, filtered when calculating final results.
        self.gt = self.test_loader.dataset.y

        # thresholds for visualization
        self.th_q95 = np.quantile(self.train_anoscs, 0.95)
        self.th_q99 = np.quantile(self.train_anoscs, 0.99)
        self.th_q100 = np.quantile(self.train_anoscs, 1.00)
        # self.th_off_f1_best = get_best_static_threshold(gt=self.gt, anomaly_scores=self.test_anoscs)


    def infer(self, mode, cols):
        result_df = pd.DataFrame(columns=cols)
        gt = self.test_loader.dataset.y

        # for single inference: select specific threshold tau
        th = self.args.thresholding
        if th[0] == "q":
            th = float(th[1:]) / 100
            tau = np.quantile(self.train_anoscs, th)
        elif th == "off_f1_best":
            tau = self.th_off_f1_best
        else:
            raise ValueError(f"Thresholding mode {self.args.thresholding} is not supported.")

        # get result
        if mode == "offline":
            anoscs, pred = self.offline(tau)


        elif mode == "online":
            anoscs, pred = self.online(self.test_loader, tau, normalization=self.args.normalization)
            # result = get_summary_stats(gt, pred)
            # roc_auc = calculate_roc_auc(gt, anoscs,
            #                             path=self.args.output_path,
            #                             save_roc_curve=self.args.save_roc_curve,
            #                             drop_intermediate=False,
            #                             )
            # result["ROC_AUC"] = roc_auc

            # pr_auc = calculate_pr_auc(gt, anoscs,
            #                           path=self.args.output_path,
            #                           save_pr_curve=self.args.save_pr_curve,
            #                           )
            # result["PR_AUC"] = pr_auc

            # result_df = pd.DataFrame([result], index=[mode], columns=result_df.columns)
            # result_df.at[mode, "tau"] = tau


        return result_df


    def online(self, *args):
        raise NotImplementedError()

# models
'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class Detrender(nn.Module):
    def __init__(self, num_features: int, gamma=0.99):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        """
        super(Detrender, self).__init__()
        self.num_features = num_features
        self.gamma = gamma
        self.mean = nn.Parameter(torch.zeros(1, 1, self.num_features), requires_grad=False)


    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x


    def _update_statistics(self, x):
        dim2reduce = tuple(range(0, x.ndim-1))
        mu = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.mean.lerp_(mu, 1-self.gamma)


    def _set_statistics(self, x:torch.Tensor):
        self.mean = nn.Parameter(x, requires_grad=False)


    def _normalize(self, x):
        x = x - self.mean
        return x


    def _denormalize(self, x):
        x = x + self.mean
        return x


class MLP(nn.Module):
    def __init__(self, seq_len, num_channels, latent_space_size, gamma, normalization="None"):
        super().__init__()
        self.L, self.C = seq_len, num_channels
        self.encoder = Encoder(seq_len*num_channels, latent_space_size)
        self.decoder = Decoder(seq_len*num_channels, latent_space_size)
        self.normalization = normalization

        if self.normalization == "Detrend":
            self.use_normalizer = True
            self.normalizer = Detrender(num_channels, gamma=gamma)
        else:
            self.use_normalizer = False


    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        if self.use_normalizer:
            X = self.normalizer(X, "norm")
            
        z = self.encoder(X.reshape(B, L*C))
        out = self.decoder(z).reshape(B, L, C)

        if self.use_normalizer:
            out = self.normalizer(out, "denorm")
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_space_size, input_size // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return out

# others
import torch
import torch.nn.functional as F
import numpy as np


class MLP_Trainer(Trainer):
    def __init__(self, model, train_loader, epochs=10, lr=1e-3, L2_reg=0, device='cuda'):
        super(MLP_Trainer, self).__init__(model, train_loader)

        self.model = model
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=L2_reg) # L2 Reg is set to zero by default, but can be set as needed.


    def train(self):

        train_iterator = tqdm(
            range(1, self.epochs + 1),
            total=self.epochs,
            desc="training epochs",
            leave=True
        )

        best_train_stats = None
        for epoch in train_iterator:
            train_stats = self.train_epoch()
            # self.logger.info(f"epoch {epoch} | train_stats: {train_stats}")
            # self.checkpoint(os.path.join(self.args.checkpoint_path, f"epoch{epoch}.pth"))

            # if best_train_stats is None or train_stats < best_train_stats:
            #     self.logger.info(f"Saving best results @epoch{epoch}")
            #     self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))
            #     best_train_stats = train_stats


    def train_epoch(self):
        self.model.train()
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary


    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.device)
        B, L, C = X.shape

        # recon
        Xhat = self.model(X)

        # optimize
        self.optimizer.zero_grad()
        loss = F.mse_loss(Xhat, X)
        loss.backward()
        self.optimizer.step()

        out = {
            "recon_loss": loss.item(),
            "summary": loss.item(),
        }
        return out


class MLP_Tester(Tester):
    def __init__(self, model, train_loader, test_loader, device='cuda', load=False, lr=1e-3):
        super(MLP_Tester, self).__init__(model, train_loader, test_loader)

        self.model = model
        self.device = device
        self.lr = lr

        if load:
            self.load_trained_model()
            self.prepare_stats()


    @torch.no_grad()
    def calculate_anomaly_scores(self, dataloader):
        recon_errors = self.calculate_recon_errors(dataloader) # B, L, C
        #anomaly_scores = recon_errors.mean(dim=2).reshape(-1).detach().cpu() # B, L -> (T=B*L, )
        anomaly_scores = recon_errors.mean(axis=2).reshape(-1)
        return anomaly_scores


    @torch.no_grad()
    def calculate_recon_errors(self, dataloader):
        '''
        :return:  returns (B, L, C) recon loss tensor
        '''
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="calculating reconstruction errors",
            leave=True
        )
        recon_errors = []
        Xs, Xhats = [], []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.device)
            B, L, C = X.shape
            Xhat = self.model(X)

            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_error = recon_error.detach().cpu().numpy()
            recon_errors.append(recon_error)
            torch.cuda.empty_cache()

        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors


    def online(self, dataloader, init_thr, normalization="None"):
        # self.load_trained_model() # reset

        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="inference",
            leave=True
        )

        tau = init_thr
        TT_optimizer = torch.optim.SGD([p for p in self.model.parameters()], lr=self.lr)

        Xs, Xhats = [], []
        preds = []
        As, thrs = [], []

        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.device)
            B, L, C = X.shape

            # Update of test-time statistics.
            if normalization == "Detrend":
                self.model.normalizer._update_statistics(X)

            # inference
            Xhat = self.model(X)
            E = (Xhat-X)**2
            A = E.mean(dim=2)
            # A: (B, L, C) -> (B, L)
            ytilde = (A >= tau).float()
            pred = ytilde

            # log model outputs
            Xs.append(X)
            Xhats.append(Xhat.clone().detach())
            As.append(A[:,-1].clone().detach())
            preds.append(pred.clone().detach())
            thrs.append(tau)

            # learn new-normals
            TT_optimizer.zero_grad()
            mask = (ytilde == 0)
            recon_loss = (A * mask).mean()
            recon_loss.backward()
            TT_optimizer.step()


        # outputs
        anoscs = torch.cat(As, axis=0).reshape(-1).detach().cpu().numpy()

        return anoscs
    
    

from .base import BaseDetector
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from torch.utils.data import DataLoader
from ..utils.dataset import ReconstructDataset

class M2N2(BaseDetector):
    def __init__(self, 
                 win_size=12,
                 stride=12,
                 num_channels=1, 
                 batch_size=64,
                 epochs=10,
                 latent_dim=128,
                 lr=1e-3,
                 normalization="Detrend",
                 gamma=0.99,
                 th='q95'):

        self.model_name = 'M2N2'
        self.normalization = normalization
        self.device = get_gpu(True)
        self.model = MLP(
            seq_len=win_size,
            num_channels=num_channels,
            latent_space_size=latent_dim,
            gamma=gamma,
            normalization=normalization,
        ).to(self.device)
        
        self.th = float(th[1:]) / 100
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.stride = stride
        self.validation_size = 0.1
        
        
        
        
    def fit(self, data):
        print("======================TRAIN MODE======================")

        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.trainer = MLP_Trainer(
            model=self.model,
            train_loader=train_loader,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device
        )
        
        self.trainer.train()
        
        self.tester = MLP_Tester(
            model=self.model,
            train_loader=train_loader,
            test_loader=train_loader,
            device=self.device,
            load=False,
            lr=self.lr
        )
        self.model.eval()
        
        train_anoscs = self.tester.calculate_anomaly_scores(train_loader)
        
        self.tau = np.quantile(train_anoscs, self.th)
        
        
    def decision_function(self, data):
        self.model.eval()

        print("======================TEST MODE======================")

        
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        
        tester = MLP_Tester(
            model=self.model,
            train_loader=test_loader,
            test_loader=test_loader,
            device=self.device,
            load=False,
            lr=self.lr
        )
        
        anoscs = tester.online(test_loader, self.tau, normalization=self.normalization)

        # Custom stride length
        scores_win = [anoscs[i] for i in range(anoscs.shape[0])]
        self.decision_scores_ = np.zeros(len(data))
        count = np.zeros(len(data))
        for i, score in enumerate(scores_win):
            start = i * self.stride
            end = start + self.win_size
            self.decision_scores_[start:end] += score
            count[start:end] += 1
        self.decision_scores_ = self.decision_scores_ / np.maximum(count, 1)
        
        return self.decision_scores_
        
        
        
        