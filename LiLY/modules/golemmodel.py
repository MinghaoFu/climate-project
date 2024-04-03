
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl

from torch.nn.utils import spectral_norm
from Caulimate.Utils.Tools import check_tensor
from Caulimate.Utils.Lego import CustomMLP, PartiallyPeriodicMLP

class TApproximator(nn.Module):
    def __init__(self, args, m_embed_dim, hid_dim, cos_len, periodic_ratio=0.2):
        super(TApproximator, self).__init__()
        self.m_embedding = nn.Embedding(12, m_embed_dim)
        self.m_encoder = CustomMLP(m_embed_dim, hid_dim, 1, 3)
        
        self.t_encoder = PartiallyPeriodicMLP(1, hid_dim, 1, cos_len, periodic_ratio)
        
        self.fc = CustomMLP(2*hid_dim, hid_dim, 1, 3)
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, t):
        m = (t % 12).to(torch.int64).squeeze(dim=1)
        m_embed = self.m_embedding(m)
        t_embed = t
        xm = self.m_encoder(m_embed)
        xt = self.t_encoder(t_embed)
        
        return xt + xm 

class GolemModel(pl.LightningModule):
    def __init__(self, args, d, in_dim=1, equal_variances=True,
                 seed=1, B_init=None):
        super().__init__()
        self.save_hyperparameters()
        self.d = d
        self.seed = seed
        self.batch_size = args.batch_size
        self.equal_variances = equal_variances
        self.B_init = B_init
        self.in_dim = in_dim
        self.embedding_dim = args.embedding_dim
        self.num = args.num
        self.distance = args.distance
        self.tol = args.tol

        self.B_lags = []
        self.lag = args.lag
        
        self.gradient = []
        self.Bs = np.empty((0, d, d))
        self.TApproximators = nn.ModuleList()

        for _ in range(self.d ** 2 - self.d - (self.d - self.distance) * (self.d - self.distance - 1)):
            self.TApproximators.append(TApproximator(args, 2, 32, args.cos_len, periodic_ratio=0.1))

    def decompose_t_batch(self, t_batch):
        a_batch = t_batch // 100
        b_batch = t_batch % 100
        return a_batch, b_batch
    
    def apply_spectral_norm(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                spectral_norm(module)
        
    def generate_B(self, T):
        #T_embed = (1 / self.alpha) * torch.cos(self.beta * T + self.bias)
        
        T_embed = T
        B = []
        for layer in self.TApproximators:
            B_i = layer(T)
            B_i_sparse = B_i.masked_fill_(torch.abs(B_i) < self.tol, 0)
            B.append(B_i_sparse)
        B = torch.cat(B, dim=1)
    
        B = self.reshape_B(B)
        return B, T_embed
        
    def _preprocess(self, B):
        B = B.clone()
        B_shape = B.shape
        if len(B_shape) == 3:  # Check if B is a batch of matrices
            for i in range(B_shape[0]):  # Iterate over each matrix in the batch
                B[i].fill_diagonal_(0)
        else:
            print("Input tensor is not a batch of matrices.")
            B.data.fill_diagonal_(0)
        return B

    
    def forward_latent(self, X, T, B_init=None, init_f=False, B_label = None):
        B, T_embed = self.generate_B(T)
        self.Bs = np.concatenate((self.Bs, B.detach().cpu().numpy()), axis=0)
        
        return B
        
    def forward(self, T, B_label=None):
        B, T_embed = self.generate_B(T)
        if self.training is False:
            self.Bs = np.concatenate((self.Bs, B.detach().cpu().numpy()), axis=0)
        else:
            self.Bs = self.Bs[:0]
        return B
        # B, T_embed = self.generate_B(T)
        # if init_f:
        #     if B_label is not None:
        #         label = B_label.reshape(B_label.shape[0], self.d, self.d)
        #     else:
        #         label = check_tensor(B_init).repeat(B.shape[0], 1, 1)
            
        # self.Bs.append(B.detach().cpu().numpy())
        
        # if init_f:
        #     losses = {'total_loss': torch.nn.functional.mse_loss(B, label)}
        #     return losses, B 
        # else:
        #     losses = self.compute_loss(X, B, T_embed) 
        #     return losses
        
    def training_step(self, batch, batch_idx):
        X, T, Bs = batch
        B_est = self.forward(T)
        losses = self.compute_loss(X, T, B_est)
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        X, T = batch
        B = self.forward(T)
        
        losses = self.compute_loss(X, B, T)
        return losses['total_loss']
    
    def reshape_B(self, B):
        B_zeros = check_tensor(torch.zeros(B.shape[0], self.d, self.d))
        idx = 0
        for i in range(self.d ** 2):
            row = i // self.d
            col = i % self.d
            if -self.distance <= col - row <= self.distance and row != col:
                B_zeros[:, row, col] = B[:, idx]
                idx += 1
            else:
                continue
        return B_zeros
    
    def mask_B(self, B, fix: bool):
        if fix:
            mask = check_tensor(torch.zeros(B.shape[0], self.d, self.d))

            indices = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 2), (4, 3)]
            for i, j in indices:
                mask[:, i, j] = 1.0
            masked_data = B * mask
            return masked_data
        else:
            l_mask = self.d - self.distance - 1
            mask_upper = torch.triu(torch.zeros((self.d, self.d)), diagonal=1)
            mask_upper[:l_mask, -l_mask:] = torch.triu(torch.ones((l_mask, l_mask)), diagonal=0)

            mask_lower = torch.tril(torch.zeros((self.d, self.d)), diagonal=-1)
            mask_lower[-l_mask:, :l_mask] = torch.tril(torch.ones((l_mask, l_mask)), diagonal=0)
            mask = mask_upper + mask_lower
            mask = mask.expand(self.batch_size, self.d, self.d)
            B = B * check_tensor(1 - mask)
            for i in range(self.batch_size):
                B[i] = top_k_abs_tensor(B[i], 6)
            return B

    def compute_loss(self, X, T, B, B_label=None):
        if B_label is not None:
            if self.args.sparse:
                total_loss = ((B - B_label) ** 2).coalesce().values().sum()
            else:
                total_loss = torch.nn.functional.mse_loss(B, B_label)
            losses = {'total_loss': total_loss}
            return losses
        else:
            batch_size = X.shape[0]
            losses = {}
            total_loss = 0
            X = X - X.mean(axis=0, keepdim=True)
            likelihood = torch.sum(self._compute_likelihood(X, B)) / batch_size
            
            for l in self.args.loss.keys():
                if l == 'L1':
                    #  + torch.sum(self._compute_L1_group_penalty(B))
                    losses[l] = self.args.loss[l] * (torch.sum(self._compute_L1_penalty(B))) / batch_size
                    total_loss += losses[l]
                elif l == 'dag':
                    losses[l] = self.args.loss[l] * torch.sum(self._compute_h(B)) / batch_size
                    total_loss += losses[l]
                elif l == 'grad':
                    losses[l] = self.args.loss[l] * torch.sum(self._compute_gradient_penalty(B, T)) / batch_size
                    total_loss += losses[l]
                elif l == 'flat':
                    losses[l] = self.args.loss[l] * torch.sum(torch.pow(B[:, 1:] - B[:, :-1], 2)) / batch_size
                    total_loss += losses[l]
            
            losses['likelihood'] = likelihood
            losses['total_loss'] = total_loss + likelihood
            #self.gradient.append(self._compute_gradient_penalty(losses['total_loss']).cpu().detach().item())

            return losses
        
    def _compute_likelihood(self, X, B):
        if self.args.sparse:
            import pdb; pdb.set_trace()
            BX = torch.sparse.mm(B, X)
            nnz = self.args.indices.size
            v = check_tensor(torch.tensor([1] * nnz, dtype=torch.int64))
            i = check_tensor(torch.tensor([[i for i in range(nnz)], [i for i in range(nnz)]]))
            I = torch.sparse_coo_tensor(i, v, BX.shape)
        else:
            X = X.unsqueeze(2)
            if self.args.equal_variances:
                return 0.5 * self.args.d_X * torch.log(
                    torch.square(
                        torch.linalg.norm(X - B @ X)
                    )
                ) - torch.linalg.slogdet(check_tensor(torch.eye(self.args.d_X)) - B)[1]
            else:
                return 0.5 * torch.sum(
                    torch.log(
                        torch.sum(
                            torch.square(X - B @ X), dim=0
                        )
                    )
                ) - torch.linalg.slogdet(check_tensor(torch.eye(self.args.d_X)) - B)[1]

    def _compute_L1_penalty(self, B):
        return torch.norm(B, p=1, dim=(-2, -1)) 
   
    def _compute_L1_group_penalty(self, B):
        return torch.norm(B, p=2, dim=(0))

    def _compute_h(self, B):
        matrix_exp = torch.exp(torch.abs(torch.matmul(B, B)))
        traces = torch.sum(torch.diagonal(matrix_exp, dim1=-2, dim2=-1), dim=-1) - B.shape[1]
        return traces

    def _compute_smooth_penalty(self,B_t):
        B = B_t.clone().data
        batch_size = B.shape[0]
        for i in range(batch_size):
            b_fft = torch.fft.fft2(B[i])
            b_fftshift = torch.fft.fftshift(b_fft)
            center_idx = b_fftshift.shape[0] // 2
            b_fftshift[center_idx, center_idx] = 0.0
            b_ifft = torch.fft.ifft2(torch.fft.ifftshift(b_fftshift))
            B[i] = b_ifft
            
        return torch.norm(B, p=1, dim=(-2, -1))
    
    def _compute_gradient_penalty(self, loss):
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm1 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        gradients = torch.autograd.grad(outputs=loss, inputs=self.linear1.parameters(), retain_graph=True)
        gradient_norm2 = torch.sqrt(sum((grad**2).sum() for grad in gradients))
        
        return gradient_norm1 + gradient_norm2