from Caulimate.Data.SimLinGau import SimLinGau
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Tools import check_array, check_tensor, makedir, linear_regression_initialize, load_yaml
from Caulimate.Utils.Visualization import save_DAG
import sys
sys.path.append('..')
from LiLY.modules.golemmodel import GolemModel
import torch
import torch.nn as nn
import torch.optim as optim
import os, pwd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb
import pytorch_lightning as fpl
import wandb
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from tqdm import tqdm


class golem_loss(nn.Module):
    def __init__(self, args):
        super(golem_loss, self).__init__()
        self.args = args
        
    def forward(self, X, T, B, B_label=None):
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
        
    def sparse_matmul(self, B, X):
        import pdb; pdb.set_trace()
        inds = check_tensor(self.args.indices).unsqueeze(0).expand(B.shape[0], -1, -1)
        BX = X.new(X.shape)
        BX = torch.gather(X, dim=2, index=inds)
        
        return BX
        
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

if __name__ == '__main__':
    args = load_yaml('/home/minghao.fu/workspace/climate/LiLY/configs/golem.yaml')
    args.save_dir = os.path.join(args.save_dir, f'{args.dataset}_{args.d_X}_{args.distance}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))#, save_dir=log_dir)
    
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GolemModel(args, args.d_X, in_dim=1, equal_variances=True, seed=1,)
    
    if args.dataset == 'synthetic':
        dataset_name = f'golem_{args.graph_type}_{args.degree}_{args.noise_type}_{args.num}_{args.d_X}_{args.cos_len}'
        if args.load_data == True:
            # X = np.load(f'./dataset/{dataset_name}/X.npy')
            # Bs = np.load(f'./dataset/{dataset_name}/Bs.npy')
            dataset = SimLinGau(args.num, args.noise_type, args.d_X, args.degree, args.cos_len, args.save_dir, args.seed, vary_func=np.cos)
            X, Bs, B_bin, coords = dataset.X, dataset.Bs, dataset.B_bin, dataset.coords
            # X = data['X']
            # Bs = data['Bs']
            # B_bin = data['B_bin']
            # coords = data['coords']
        else:
            pass
            # B_scale = 0.1
            # B_ranges = ((B_scale * -2.0, B_scale * -0.5),
            #                     (B_scale * 0.5, B_scale * 2.0))
            # Bs, B_bin = simulate_time_vary_weight(args.d_X, args.num, args.cos_len, B_ranges, args.degree, args.graph_type, args.vary_type, args.seed)
            # Bs = check_array(Bs)

            # X = np.zeros((args.num, args.d_X))
            # for i in range(args.num):
            #     X[i:i+1, :] = simulate_graph.simulate_linear_sem(Bs[i], 1, args.noise_type, rs)
            # #self.X = np.matmul(self.X, np.linalg.inv(np.eye(self.d_X) - self.Bs))
            
            # makedir('./dataset/{}'.format(dataset_name), remove_exist=True)
            # np.save(f'./dataset/{dataset_name}/X.npy', X)
            # np.save(f'./dataset/{dataset_name}/Bs.npy', Bs)
    # elif args.dataset == 'CESM': 
    #     X, Bs = CESM_dataset.load_data(args)
        
                
    X = check_tensor(X, dtype=torch.float32)
    X = X - X.mean(dim=0)
    T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
    Bs_gt = check_tensor(Bs, dtype=torch.float32)
    tensor_dataset = TensorDataset(X, T, Bs_gt)
    data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    golem_loss = golem_loss(args)
    
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f'--- Load checkpoint from {args.checkpoint_path}')
    else:
        B_init = linear_regression_initialize(X, args.distance)
        B_init = check_tensor(B_init, dtype=torch.float32)
        if args.regression_init:
            for epoch in tqdm(range(args.init_epoch)):
                for batch_X, batch_T, batch_B in data_loader:
                    optimizer.zero_grad()  
                    B_pred = model(T)
                    B_label = check_tensor(B_init).repeat(batch_T.shape[0], 1, 1)
                    loss = golem_loss(batch_X, batch_T, B_pred, B_init)
                    loss['total_loss'].backward()
                    optimizer.step()

            #save_epoch_log(args, model, B_init, X, T, -1)
            print(f"--- Init F based on linear regression, ultimate loss: {loss['total_loss'].item()}")
        
    for epoch in range(args.epoch):
        model.train()
        for X_batch, T_batch, B_batch in data_loader:
            optimizer.zero_grad()
            B_pred = model(T_batch)
            losses = golem_loss(X_batch, T_batch, B_pred)
            losses['total_loss'].backward()
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
    
            optimizer.step()
            
        if epoch % 100 == 0:
            model.eval()
            Bs_pred = check_array(model(T).permute(0, 2, 1))
            # if args.dataset != 'synthetic':
            #     Bs_gt = Bs_pred
            #     for i in range(args.num):
            #         Bs_gt[i] = postprocess(Bs_gt[i])
            if args.dataset != 'synthetic':
                Bs_gt = None
            save_DAG(args.num, args.save_dir, epoch, Bs_pred, Bs_gt, graph_thres=args.graph_thres, add_value=False)
            print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}', 'checkpoint.pth'))
            
            # pred_B = model(T)
            # print(pred_B[0], dataset.B)
            # fig = plot_solutions([B_gt.T, B_est, W_est, M_gt.T, M_est], ['B_gt', 'B_est', 'W_est', 'M_gt', 'M_est'], add_value=True, logger=self.logger)
            # self.logger.experiment.log({"Fig": [wandb.Image(fig)]})