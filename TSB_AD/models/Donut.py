"""
This function is adapted from [donut] by [haowen-xu]
Original source: [https://github.com/NetManAIOps/donut]
"""

from typing import Dict
import numpy as np
import torchinfo
import torch
from torch import nn, optim
import tqdm
import os, math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Sequence, Union, Callable

from ..utils.torch_utility import EarlyStoppingTorch, get_gpu
from ..utils.dataset import ReconstructDataset    

class DonutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, mask_prob) -> None:
        super().__init__()

        """
        Xu2018

        :param input_dim: Should be window_size * features
        :param hidden_dims:
        :param latent_dim:
        """

        self.latent_dim = latent_dim
        self.mask_prob = mask_prob
        
        encoder = VaeEncoder(input_dim, hidden_dim, latent_dim)
        decoder = VaeEncoder(latent_dim, hidden_dim, input_dim)
        
        self.vae = VAE(encoder=encoder, decoder=decoder, logvar_out=False)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x = inputs
        B, T, D = x.shape

        if self.training:
            # Randomly mask some inputs
            mask = torch.empty_like(x)
            mask.bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        # Run the VAE
        x = x.view(x.shape[0], -1)  
        mean_z, std_z, mean_x, std_x, sample_z = self.vae(x, return_latent_sample=True)

        # Reshape the outputs
        mean_x = mean_x.view(B, T, D)
        std_x = std_x.view(B, T, D)
        return mean_z, std_z, mean_x, std_x, sample_z, mask

def sample_normal(mu: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False, num_samples: int = 1):
    # ln(σ) = 0.5 * ln(σ^2) -> σ = e^(0.5 * ln(σ^2))
    if log_var:
        sigma = std_or_log_var.mul(0.5).exp_()
    else:
        sigma = std_or_log_var

    if num_samples == 1:
        eps = torch.randn_like(mu)  # also copies device from mu
    else:
        eps = torch.rand((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
    # z = μ + σ * ϵ, with ϵ ~ N(0,I)
    return eps.mul(sigma).add_(mu)

def normal_standard_normal_kl(mean: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        kl_loss = torch.sum(1 + std_or_log_var - mean.pow(2) - std_or_log_var.exp(), dim=-1)
    else:
        kl_loss = torch.sum(1 + torch.log(std_or_log_var.pow(2)) - mean.pow(2) - std_or_log_var.pow(2), dim=-1)
    return -0.5 * kl_loss
    

def normal_normal_kl(mean_1: torch.Tensor, std_or_log_var_1: torch.Tensor, mean_2: torch.Tensor,
                     std_or_log_var_2: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        return 0.5 * torch.sum(std_or_log_var_2 - std_or_log_var_1 + (torch.exp(std_or_log_var_1)
                               + (mean_1 - mean_2)**2) / torch.exp(std_or_log_var_2) - 1, dim=-1)

    return torch.sum(torch.log(std_or_log_var_2) - torch.log(std_or_log_var_1) \
                     + 0.5 * (std_or_log_var_1**2 + (mean_1 - mean_2)**2) / std_or_log_var_2**2 - 0.5, dim=-1)


class VAELoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', logvar_out: bool = True):
        super(VAELoss, self).__init__(size_average, reduce, reduction)
        self.logvar_out = logvar_out

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        z_mean, z_std_or_log_var, x_dec_mean, x_dec_std = predictions[:4]
        if len(predictions) > 4:
            z_prior_mean, z_prior_std_or_logvar = predictions[4:]
        else:
            z_prior_mean, z_prior_std_or_logvar = None, None

        y, = targets

        # Gaussian nnl loss assumes multivariate normal with diagonal sigma
        # Alternatively we can use torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)
        # or torch.distribution.MultivariateNormal(mean, cov).log_prob(y).sum(-1)
        # with cov = torch.eye(feat_dim).repeat([1,bz,1,1])*std.pow(2).unsqueeze(-1).
        # However setting up a distribution seems to be an unnecessary computational overhead.
        # However, this requires pytorch version > 1.9!!!
        nll_gauss = F.gaussian_nll_loss(x_dec_mean, y, x_dec_std.pow(2), reduction='none').sum(-1)
        # For pytorch version < 1.9 use:
        # nll_gauss = -torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)

        # get KL loss
        if z_prior_mean is None and z_prior_std_or_logvar is None:
            # If a prior is not given, we assume standard normal
            kl_loss = normal_standard_normal_kl(z_mean, z_std_or_log_var, log_var=self.logvar_out)
        else:
            if z_prior_mean is None:
                z_prior_mean = torch.tensor(0, dtype=z_mean.dtype, device=z_mean.device)
            if z_prior_std_or_logvar is None:
                value = 0 if self.logvar_out else 1
                z_prior_std_or_logvar = torch.tensor(value, dtype=z_std_or_log_var.dtype, device=z_std_or_log_var.device)

            kl_loss = normal_normal_kl(z_mean, z_std_or_log_var, z_prior_mean, z_prior_std_or_logvar,
                                       log_var=self.logvar_out)

        # Combine
        final_loss = nll_gauss + kl_loss

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)


class MaskedVAELoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedVAELoss, self).__init__(size_average, reduce, reduction, logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            mean_z = mean_z.unsqueeze(1)
            std_z = std_z.unsqueeze(1)
            return super(MaskedVAELoss, self).forward((mean_z, std_z, mean_x, std_x), (actual_x,), *args, **kwargs)

        # If the loss is masked, one of the terms in the kl loss is weighted, so we can't compute it exactly
        # anymore and have to use a MC approximation like for the output likelihood
        nll_output = torch.sum(mask * F.gaussian_nll_loss(mean_x, actual_x, std_x**2, reduction='none'), dim=-1)

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes p(z) = N(z| 0, I), we drop constants
        beta = torch.mean(mask, dim=(1, 2)).unsqueeze(-1)
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1, keepdim=True)

        nll_approx = torch.sum(F.gaussian_nll_loss(mean_z, sample_z, std_z**2, reduction='none'), dim=-1, keepdim=True)

        final_loss = nll_output + nll_prior - nll_approx

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)

class MLP(torch.nn.Module):
    def __init__(self, input_features: int, hidden_layers: Union[int, Sequence[int]], output_features: int,
                 activation: Callable = torch.nn.Identity(), activation_after_last_layer: bool = False):
        super(MLP, self).__init__()

        self.activation = activation
        self.activation_after_last_layer = activation_after_last_layer

        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        layers = [input_features] + list(hidden_layers) + [output_features]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(inp, out) for inp, out in zip(layers[:-1], layers[1:])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.activation(out)

        out = self.layers[-1](out)
        if self.activation_after_last_layer:
            out = self.activation(out)

        return out

class VaeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VaeEncoder, self).__init__()
        
        self.latent_dim = latent_dim

        self.mlp = MLP(input_dim, hidden_dim, 2*latent_dim, activation=torch.nn.ReLU(), activation_after_last_layer=False)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        mlp_out = self.mlp(x)

        mean, std = mlp_out.tensor_split(2, dim=-1)
        std = self.softplus(std)

        return mean, std
    
class VAE(torch.nn.Module):
    """
    VAE Implementation that supports normal distribution with diagonal cov matrix in the latent space
    and the output
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, logvar_out: bool = True):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.log_var = logvar_out

    def forward(self, x: torch.Tensor, return_latent_sample: bool = False, num_samples: int = 1,
                force_sample: bool = False) -> Tuple[torch.Tensor, ...]:
        z_mu, z_std_or_log_var = self.encoder(x)

        if self.training or num_samples > 1 or force_sample:
            z_sample = sample_normal(z_mu, z_std_or_log_var, log_var=self.log_var, num_samples=num_samples)
        else:
            z_sample = z_mu

        x_dec_mean, x_dec_std = self.decoder(z_sample)

        if not return_latent_sample:
            return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std

        return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std, z_sample



class Donut():
    def __init__(self,
                 win_size=120,
                 input_c=1,
                 batch_size=128,     # 32, 128
                 grad_clip=10.0,
                 num_epochs=50,
                 mc_samples=1024,
                 hidden_dim=100,
                 latent_dim=8,
                 inject_ratio=0.01,
                 lr=1e-4,
                 l2_coff=1e-3,
                 patience=3,
                 validation_size=0):
        super().__init__()
        self.__anomaly_score = None
        
        self.cuda = True
        self.device = get_gpu(self.cuda)
        
        self.win_size = win_size
        self.input_c = input_c
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.num_epochs = num_epochs
        self.mc_samples = mc_samples
        self.validation_size = validation_size
        
        input_dim = self.win_size*self.input_c
        
        self.model = DonutModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, mask_prob=inject_ratio).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=l2_coff)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
        self.vaeloss = MaskedVAELoss()
        
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=patience)
        
    def train(self, train_loader, epoch):
        self.model.train(mode=True)
        avg_loss = 0
        loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
        for idx, (x, target) in loop:
            x, target = x.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # print('x: ', x.shape)
            
            output = self.model(x)
            loss = self.vaeloss(output, (target,))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            avg_loss += loss.cpu().item()
            loop.set_description(f'Training Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
        
        return avg_loss/max(len(train_loader), 1)
                
    def valid(self, valid_loader, epoch):
        self.model.eval()
        avg_loss = 0
        loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                loss = self.vaeloss(output, (target,))
                avg_loss += loss.cpu().item()
                loop.set_description(f'Validation Epoch [{epoch}/{self.num_epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                
        return avg_loss/max(len(valid_loader), 1)
        
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
                    
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train(train_loader, epoch)
            if len(valid_loader) > 0:
                valid_loss = self.valid(valid_loader, epoch)
            self.scheduler.step()
            
            if len(valid_loader) > 0:
                self.early_stopping(valid_loss, self.model)
            else:
                self.early_stopping(train_loss, self.model)
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
            for idx, (x, _) in loop:
                x = x.to(self.device)
                x_vae = x.view(x.shape[0], -1)
                B, T, D = x.shape

                res = self.model.vae(x_vae, return_latent_sample=False, num_samples=self.mc_samples)
                z_mu, z_std, x_dec_mean, x_dec_std = res

                x_dec_mean = x_dec_mean.view(self.mc_samples, B, T, D)
                x_dec_std = x_dec_std.view(self.mc_samples, B, T, D)                
                nll_output = torch.sum(F.gaussian_nll_loss(x_dec_mean[:, :, -1, :], x[:, -1, :].unsqueeze(0),
                                                   x_dec_std[:, :, -1, :]**2, reduction='none'), dim=(0, 2))
                nll_output /= self.mc_samples


                scores.append(nll_output.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        
        assert scores.ndim == 1
        
        import shutil
        if self.save_path and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
            
        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return super().get_y_hat
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.win_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
    