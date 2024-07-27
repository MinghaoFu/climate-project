"""Temporal VAE with gaussian margial and laplacian transition prior"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pytorch_lightning as pl
import torch.distributions as D
from torch.nn import functional as F
from .components.beta import BetaVAE_MLP, BetaTVAE_MLP
from .components.transition import NPChangeTransitionPrior, NPTransitionPrior, NPStatePrior
from .components.mlp import MLPEncoder, MLPDecoder, Inference
from .components.flow import ComponentWiseCondSpline
from .metrics.correlation import compute_mcc
from Caulimate.Utils.Tools import check_tensor
from Caulimate.Utils.Lego import PartiallyPeriodicMLP
import ipdb as pdb

class ModularShifts(pl.LightningModule):

    def __init__(
        self, 
        input_dim,
        length,
        obs_dim,
        dyn_dim, 
        lag,
        nclass,
        hidden_dim=128,
        dyn_embedding_dim=2,
        obs_embedding_dim=2,
        trans_prior='NP',
        lr=1e-4,
        infer_mode='F',
        bound=5,
        count_bins=8,
        order='linear',
        beta=0.0025,
        gamma=0.0075,
        sigma=0.0025,
        decoder_dist='gaussian',
        correlation='Pearson'):
        '''Nonlinear ICA for general causal processes with modualar distribution shifts'''
        super().__init__()
        # Transition prior must be L (Linear), NP (Nonparametric)
        self.save_hyperparameters() # save hyperparameters to checkpoint
        
        assert trans_prior in ('L', 'NP')
        self.obs_dim = obs_dim
        self.dyn_dim = dyn_dim
        self.obs_embedding_dim = obs_embedding_dim
        self.dyn_embedding_dim = dyn_embedding_dim
        self.z_dim = obs_dim + dyn_dim
        self.lag = lag
        self.input_dim = input_dim
        self.lr = lr
        self.lag = lag
        self.length = length
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.correlation = correlation
        self.decoder_dist = decoder_dist
        self.infer_mode = infer_mode
        # Domain embeddings (dynamics)
        self.dyn_embed_func = nn.Embedding(nclass, dyn_embedding_dim)
        self.obs_embed_func = nn.Embedding(nclass, obs_embedding_dim)
        # Flow for nonstationary regimes
        self.flow = ComponentWiseCondSpline(input_dim=self.obs_dim,
                                            context_dim=obs_embedding_dim,
                                            bound=bound,
                                            count_bins=count_bins,
                                            order=order)
        # Factorized inference
        self.t_embed_dim = 2
        # self.net = BetaTVAE_MLP(input_dim=input_dim, 
        #                         z_dim=self.z_dim, 
        #                         t_embed_dim=self.t_embed_dim, 
        #                         hidden_dim=hidden_dim)
        
        # self.xs_net = BetaTVAE_MLP(input_dim=input_dim, 
        #                         z_dim=input_dim, 
        #                         t_embed_dim=self.t_embed_dim,
        #                         hidden_dim=hidden_dim)

        self.net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=self.z_dim, 
                                hidden_dim=hidden_dim)
        
        self.xs_net = BetaVAE_MLP(input_dim=input_dim, 
                                z_dim=input_dim, 
                                hidden_dim=hidden_dim)

        # Initialize transition prior
        if trans_prior == 'L':
            pass
            # self.transition_prior = MBDTransitionPrior(lags=lag, 
            #                                            latent_size=self.dyn_dim, 
            #                                            bias=False)
        elif trans_prior == 'NP':
            self.transition_prior = NPChangeTransitionPrior(lags=lag, 
                                                            latent_size=self.dyn_dim,
                                                            embedding_dim=dyn_embedding_dim,
                                                            num_layers=4, 
                                                            hidden_dim=hidden_dim)
            
        # self.np_gen = NPChangeTransitionPrior(lags=1,
        #                                         latent_size=self.obs_dim + self.dyn_dim, # change to observed size
        #                                         embedding_dim=dyn_embedding_dim,
        #                                         num_layers=4,
        #                                         hidden_dim=hidden_dim)
        self.t_embedding = PartiallyPeriodicMLP(1, 4, self.t_embed_dim, 500)
        self.state_prior = NPStatePrior(lags=1,
                                             latent_size=self.z_dim,
                                             input_dim=input_dim,
                                             num_layers=4,
                                             hidden_dim=hidden_dim)
        


        # base distribution for calculation of log prob under the model
        self.register_buffer('dyn_base_dist_mean', torch.zeros(self.dyn_dim))
        self.register_buffer('dyn_base_dist_var', torch.eye(self.dyn_dim))
        self.register_buffer('obs_base_dist_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_base_dist_var', torch.eye(self.obs_dim))
        self.register_buffer('sta_base_dist_mean', torch.zeros(self.input_dim))
        self.register_buffer('sta_base_dist_var', torch.eye(self.input_dim))

    @property
    def dyn_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.dyn_base_dist_mean, self.dyn_base_dist_var)

    @property
    def obs_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.obs_base_dist_mean, self.obs_base_dist_var)
    
    @property
    def sta_base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.sta_base_dist_mean, self.sta_base_dist_var)
    
    def preprocess(self, B):
        return B - B.diagonal().diag()
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def forward(self, batch):
        x, z, c = batch['xt'], batch['zt'], batch['ct']
        batch_size, length, _ = x.shape
        x_flat = x.view(-1, self.input_dim)
        _, mus, logvars, zs = self.net(x_flat)
        return zs, mus, logvars       

    def training_step(self, batch, batch_idx):
        x, z, c, h = batch['xt'], batch['zt'], batch['ct'], batch['ht']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape

        t_embedding = self.t_embedding(h / 100000).unsqueeze(1).repeat(1, length, 1)    
        x_flat = x.view(-1, self.input_dim)
        
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)

        #x_recon[:, :, 1] = x_recon[:, :, 1] + x_recon[:, :, 0] * self.t_embedding(h)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        ### Dynamics parts ###
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], dyn_embeddings)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                              torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
        log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
        log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
        kld_obs = log_qz_obs - log_pz_obs
        kld_obs = kld_obs.mean()      

        # state transition
        x_recon_s, s_mus, s_logvars, xs = self.xs_net(x_flat)

        std = s_logvars.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        x_recon_s = s_mus + std*eps

        x_recon_s = x_recon_s.view(batch_size, length, self.input_dim)
        s_mus = s_mus.reshape(batch_size, length, self.input_dim)
        s_logvars  = s_logvars.reshape(batch_size, length, self.input_dim)
        xs = xs.reshape(batch_size, length, self.input_dim)
        # import pdb; pdb.set_trace()
        # for i in range(0, length):
        s_q_dist = D.Normal(s_mus, torch.exp(s_logvars / 2))
        log_qs = s_q_dist.log_prob(xs)
        residuals, logabsdet = self.state_prior(zs, xs)
        log_ps = torch.sum(self.sta_base_dist.log_prob(residuals), dim=1) + logabsdet
        s_kld_dyn = torch.sum(torch.sum(log_qs, dim=-1), dim=-1) - log_ps
        s_kld_dyn = s_kld_dyn.mean()


        # using reconstruction by state
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon_s[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon_s[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + 0.03 * s_kld_dyn

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["zt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        self.log("train_mcc", mcc)
        self.log("train_elbo_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kld_normal", past_kld_dyn)
        self.log("train_kld_dynamics", future_kld_dyn)
        self.log("train_kld_observation", kld_obs)
        self.log("train_kld_state", s_kld_dyn)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, z, c, h = batch['xt'], batch['zt'], batch['ct'], batch['ht']
        c = torch.squeeze(c).to(torch.int64)
        batch_size, length, _ = x.shape

        t_embedding = self.t_embedding(h / 100000).unsqueeze(1).repeat(1, length, 1)    
        x_flat = x.view(-1, self.input_dim)
        
        dyn_embeddings = self.dyn_embed_func(c)
        obs_embeddings = self.obs_embed_func(c)
        obs_embeddings = obs_embeddings.reshape(batch_size,1,self.obs_embedding_dim).repeat(1,length,1)
        # Inference
        x_recon, mus, logvars, zs = self.net(x_flat)

        # Reshape to time-series format
        x_recon = x_recon.view(batch_size, length, self.input_dim)

        #x_recon[:, :, 1] = x_recon[:, :, 1] + x_recon[:, :, 0] * self.t_embedding(h)
        mus = mus.reshape(batch_size, length, self.z_dim)
        logvars  = logvars.reshape(batch_size, length, self.z_dim)
        zs = zs.reshape(batch_size, length, self.z_dim)

        # VAE ELBO loss: recon_loss + kld_loss
        
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        ### Dynamics parts ###
        # Past KLD <=> N(0,1) #
        p_dist = D.Normal(torch.zeros_like(mus[:,:self.lag,:self.dyn_dim]), torch.ones_like(logvars[:,:self.lag, :self.dyn_dim]))
        log_pz_past = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag,:self.dyn_dim]),dim=-1),dim=-1)
        log_qz_past = torch.sum(torch.sum(log_qz[:,:self.lag,:self.dyn_dim],dim=-1),dim=-1)
        past_kld_dyn = log_qz_past - log_pz_past
        past_kld_dyn = past_kld_dyn.mean()
        # Future KLD #
        log_qz_future = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs[:,:,:self.dyn_dim], dyn_embeddings)
        log_pz_future = torch.sum(self.dyn_base_dist.log_prob(residuals), dim=1) + logabsdet
        future_kld_dyn = (torch.sum(torch.sum(log_qz_future,dim=-1),dim=-1) - log_pz_future) / (length-self.lag)
        future_kld_dyn = future_kld_dyn.mean()

        ### Observation parts ###
        p_dist_obs = D.Normal(obs_embeddings[:,:,0].reshape(batch_size, length, 1), 
                              torch.exp(obs_embeddings[:,:,1].reshape(batch_size, length, 1) / 2) )
        log_pz_obs = torch.sum(torch.sum(p_dist_obs.log_prob(zs[:,:,self.dyn_dim:]), dim=1),dim=-1)
        log_qz_obs = torch.sum(torch.sum(log_qz[:,:self.lag,self.dyn_dim:],dim=-1),dim=-1)
        kld_obs = log_qz_obs - log_pz_obs
        kld_obs = kld_obs.mean()      

        # state transition
        x_recon_s, s_mus, s_logvars, xs = self.xs_net(x_flat)

        x_recon_s = x_recon_s.view(batch_size, length, self.input_dim)
        s_mus = s_mus.reshape(batch_size, length, self.input_dim)
        s_logvars  = s_logvars.reshape(batch_size, length, self.input_dim)
        xs = xs.reshape(batch_size, length, self.input_dim)
        # import pdb; pdb.set_trace()
        # for i in range(0, length):
        s_q_dist = D.Normal(s_mus, torch.exp(s_logvars / 2))
        log_qs = s_q_dist.log_prob(xs)
        residuals, logabsdet = self.state_prior(zs, xs)
        log_ps = torch.sum(self.sta_base_dist.log_prob(residuals), dim=1) + logabsdet
        s_kld_dyn = torch.sum(torch.sum(log_qs, dim=-1), dim=-1) - log_ps
        s_kld_dyn = s_kld_dyn.mean()

        # using reconstruction by state
        recon_loss = self.reconstruction_loss(x[:,:self.lag], x_recon_s[:,:self.lag], self.decoder_dist) + \
        (self.reconstruction_loss(x[:,self.lag:], x_recon_s[:,self.lag:], self.decoder_dist))/(length-self.lag)
        
        # VAE training
        loss = recon_loss + self.beta * past_kld_dyn + self.gamma * future_kld_dyn + self.sigma * kld_obs + 0.01 * s_kld_dyn

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()

        zt_true = batch["zt"].view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)

        self.log("val_mcc", mcc) 
        self.log("val_elbo_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kld_normal", past_kld_dyn)
        self.log("val_kld_dynamics", future_kld_dyn)
        self.log("val_kld_observation", kld_obs)
        self.log("val_kld_observation", s_kld_dyn)
        

        return loss
    
    def sample(self, n=64):
        with torch.no_grad():
            e = torch.randn(n, self.z_dim, device=self.device)
            eps, _ = self.spline.inverse(e)
        return eps

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []
