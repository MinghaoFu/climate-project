import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from Caulimate.Utils.Tools import check_tensor

class LatentLinearGaussian(LightningModule):
    def __init__(self, args, dataset) -> None:
        super(LatentLinearGaussian, self).__init__()
        if args.gt_init:
            self.B = torch.nn.Parameter(check_tensor(dataset.B, dtype=torch.float32))
            self.C = torch.nn.Parameter(check_tensor(dataset.C, dtype=torch.float32))
            self.EL_cov = torch.nn.Parameter(check_tensor(dataset.EL_cov.diagonal(), dtype=torch.float32)) # mutual independent->diagonal
            self.EX_cov = torch.nn.Parameter(check_tensor(dataset.EX_cov.diagonal(), dtype=torch.float32))
        else:    
            self.B = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, args.d_X, requires_grad=True))) 
            self.C = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, args.max_d_L, requires_grad=True)))
            self.EL_cov = torch.nn.Parameter(check_tensor(torch.randn(args.max_d_L, requires_grad=True)))
            self.EX_cov = torch.nn.Parameter(check_tensor(torch.randn(args.d_X, requires_grad=True)))
        
        self.n_params = sum(p.view(-1).size(0) for p in self.parameters()) - args.d_X
        with torch.no_grad():
            self.B.fill_diagonal_(0)
        
    def forward(self, args, X_cov):
        B = self.B
        self.est_X_cov, likelihood = self.log_gaussian_likelihood(B, self.C, torch.diag(self.EX_cov), torch.diag(self.EL_cov), args.num, X_cov)
        sparsity_M = args.sparsity_M * torch.norm(B, p=1) 
        sparsity_L = args.sparsity_L * torch.norm(self.C, p=1)
        sparsity = sparsity_M + sparsity_L 
        dag = args.DAG * torch.sum(torch.diagonal(torch.matrix_exp(B * B), dim1=-2, dim2=-1)) - args.d_X
        
        loss = {}
        loss['likelihood'] = likelihood
        loss['sparsity_M'] = sparsity_M
        loss['sparsity_L'] = sparsity_L
        loss['dag'] = dag
        loss['score'] = (torch.sum(torch.abs(B) >= args.graph_thres) + 2) * torch.log(check_tensor(torch.tensor(args.num))) + 2 * likelihood + sparsity + dag
        
        return loss
    
    def log_gaussian_likelihood(self, B, C, EX_cov, EL_cov, n, X_cov):
        d_X = X_cov.shape[0]
        
        I = check_tensor(torch.eye(d_X))
        inv_I_minus_B = torch.inverse(I - B)

        est_X_cov = inv_I_minus_B.T @ (C @ EL_cov @ C.t() + EX_cov) @ inv_I_minus_B

        return est_X_cov, 0.5 * (torch.slogdet(est_X_cov)[1] + \
            torch.trace(torch.mm(torch.inverse(est_X_cov), X_cov)) + \
                d_X * torch.log(check_tensor(torch.tensor(torch.pi * 2))))
    
    def training_step(self, batch, batch_idx):
        args, X_cov = batch
        loss = self(args, X_cov)
        self.log('train_loss', loss['score'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss['score']

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    