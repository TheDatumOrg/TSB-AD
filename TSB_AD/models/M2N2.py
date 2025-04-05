"""
This function is adapted from [M2N2] by [Dongmin Kim et al.]
Original source: [https://github.com/carrtesy/M2N2]
Reimplemented by: [EmorZz1G]
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDetector
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from torch.utils.data import DataLoader
from ..utils.dataset import ReconstructDataset
from typing import Literal
        

# models
'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''
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


class MLP_Trainer:
    def __init__(
            self, model, train_loader, valid_loader=None,
            epochs=10, lr=1e-3, L2_reg=0, device='cuda'
        ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr, weight_decay=L2_reg)

    def train(self):
        train_iterator = tqdm(
            range(1, self.epochs + 1),
            total=self.epochs,
            desc="training epochs",
            leave=True
        )
        if self.valid_loader is not None:
            early_stop = EarlyStoppingTorch(patience=5)
        for epoch in train_iterator:
            train_stats = self.train_epoch()
            if self.valid_loader is not None:
                valid_loss = self.valid()
                early_stop(valid_loss, self.model)
                if early_stop.early_stop:
                    break

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

    @torch.no_grad()
    def valid(self):
        assert self.valid_loader is not None, 'cannot valid without any data'
        self.model.eval()
        for i, batch_data in enumerate(self.valid_loader):
            X = batch_data[0].to(self.device)
            Xhat = self.model(X)
            loss = F.mse_loss(Xhat, X)
        return loss.item()

class MLP_Tester:
    def __init__(self, model, train_loader, test_loader, lr=1e-3, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr

    @torch.no_grad()
    def offline(self, dataloader):
        self.model.eval()
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="offline inference",
            leave=True
        )
        recon_errors = []
        for i, batch_data in enumerate(it):
            X = batch_data[0].to(self.device)
            B, L, C = X.shape
            Xhat = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none')
            recon_error = recon_error.detach().cpu().numpy()
            recon_errors.append(recon_error)
            torch.cuda.empty_cache()
        recon_errors = np.concatenate(recon_errors, axis=0) # (B, L, C)
        anomaly_scores = recon_errors.mean(axis=2).reshape(-1) # (B, L) => (B*L,)
        return anomaly_scores

    def online(self, dataloader, init_thr, normalization="None"):
        self.model.train()
        it = tqdm(
            dataloader,
            total=len(dataloader),
            desc="online inference",
            leave=True
        )
        tau = init_thr
        TT_optimizer = torch.optim.SGD(
            [p for p in self.model.parameters()], lr=self.lr)

        Xs, Xhats = [], []
        preds = []
        As, thrs = [], []
        update_count = 0
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
            # generate anomaly scores for each time step
            As.append(A.clone().detach())
            preds.append(pred.clone().detach())
            thrs.append(tau)
            # learn new-normals
            TT_optimizer.zero_grad()
            mask = (ytilde == 0)
            recon_loss = (A * mask).mean()
            recon_loss.backward()
            TT_optimizer.step()
            update_count += torch.sum(mask).item()
        # outputs
        anoscs = torch.cat(As, axis=0).reshape(-1).detach().cpu().numpy()
        print('total update count:', update_count)
        return anoscs

class M2N2(BaseDetector):
    def __init__(self, 
                 win_size=12,
                 stride=12,
                 num_channels=1, 
                 batch_size=64,
                 epochs=10,
                 latent_dim=128,
                 lr=1e-3,
                 ttlr=1e-3, # learning rate for online test-time adaptation
                 normalization="Detrend",
                 gamma=0.99,
                 th=0.95, # 95 percentile == 0.95 quantile
                 valid_size=0.2,
                 infer_mode='online'):
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
        
        self.th = th
        self.lr = lr
        self.ttlr = ttlr
        self.epochs = epochs
        self.batch_size = batch_size
        self.win_size = win_size
        self.stride = stride
        self.valid_size = valid_size
        self.infer_mode = infer_mode
        
    def fit(self, data):
        if self.valid_size is None:
            self.train_loader = DataLoader(
                dataset=ReconstructDataset(
                    data, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=True
            )
            self.valid_loader = None
        else:
            data_train = data[:int((1-self.valid_size)*len(data))]
            data_valid = data[int((1-self.valid_size)*len(data)):]
            self.train_loader = DataLoader(
                dataset=ReconstructDataset(
                    data_train, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=True
            )
            self.valid_loader = DataLoader(
                dataset=ReconstructDataset(
                    data_valid, window_size=self.win_size, stride=self.stride),
                batch_size=self.batch_size,
                shuffle=False,
            )

        self.trainer = MLP_Trainer(
            model=self.model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device
        )
        self.trainer.train()

        self.tester = MLP_Tester(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.train_loader,
            lr=self.ttlr,
            device=self.device,
        )
        train_anoscs = self.tester.offline(self.train_loader)
        self.tau = np.quantile(train_anoscs, self.th)
        print('tau', self.tau)

    def decision_function(self, data):
        self.test_loader = DataLoader(
            dataset=ReconstructDataset(
                data, window_size=self.win_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.tester = MLP_Tester(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr=self.ttlr,
            device=self.device,
        )
        if self.infer_mode == 'online':
            anoscs = self.tester.online(
                self.test_loader, self.tau,
                normalization=self.normalization)
        else:
            anoscs = self.tester.offline(self.test_loader)

        self.decision_scores_ = pad_by_edge_value(anoscs, len(data), mode='right')
        return self.decision_scores_


def pad_by_edge_value(scores, target_len, mode: Literal['both', 'left', 'right']):
    scores = np.array(scores).reshape(-1)
    assert len(scores) <= target_len, f'the length of scores is more than target one'
    print(f'origin length: {len(scores)}; target length: {target_len}')
    current_len = scores.shape[0]
    pad_total = max(target_len-current_len, 0)
    if mode == 'left':
        pad_before = pad_total
    elif mode == 'right':
        pad_before = 0
    else:
        pad_before = pad_total // 2 + 1
    pad_after = pad_total - pad_before
    padded_scores = np.pad(scores, (pad_before, pad_after), mode='edge')
    return padded_scores