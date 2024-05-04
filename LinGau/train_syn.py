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
from einops import repeat

from Caulimate.Data.SimLinGau import LinGauSuff
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Visualization import save_DAG, make_dots
from Caulimate.Utils.Tools import check_tensor, check_array, load_yaml, makedir, lin_reg_init, dict_to_class, save_log, bin_mat
from Caulimate.Utils.GraphMetric import count_graph_accuracy


args = {
    'noise_type': 'gaussian_ev',
    'load_data': True,
    'graph_type': 'ER',
    'num': 6000,
    'scale': 0.5,
    'pi': 10,
    'd_X': 10,
    'degree': 4,
    'cos_len': 1000,
    'max_eud': 70,
    'equal_variances': True,

    'train': True,
    'pretrain': False,
    'checkpoint_path': None,
    'regression_init': False,
    'loss': {
        'likelihood': 1.0,
        'L1': 1.e-2,
        'dag': 1.e-2
    },
    'reg_thres': 0.1,
    'ddp': False,
    'pre_epoch': 0,
    'epoch': 10000,
    'init_epoch': 100,
    'batch_size': 10000,
    'lag': 10,
    'synthetic': False,
    'time_varying': False,
    'sparse': False,

    'seed': 2,
    'gt_init': False,
    'embedding_dim': 5,
    'spectral_norm': False,
    'tol': 0.0,
    'graph_thres': 0.3,
    'DAG': 0.8,
    'save_dir': "./syn",

    'condition': "ignavier",
    'decay_type': "step",
    'optimizer': "ADAM",
    'weight_decay': 0.0,
    'lr': 1.e-4,
    'gradient_noise': None,
    'step_size': 1000,
    'gamma': 0.5,
    'decay': [200, 400, 800, 1000],
    'betas': [0.9, 0.999],
    'epsilon': 1.e-8,
    'momentum': 0.9
}
if __name__ == '__main__':
    args = dict_to_class(**args)
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    makedir(args.save_dir)
    save_log(os.path.join(args.save_dir, 'log.txt'), args.__str__())
    wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_data = LinGauSuff(args.num, args.d_X, args.degree, args.cos_len, args.seed, max_eud=args.max_eud, vary_func=np.cos)
    X, Bs, B_bin, coords, eud_mask = sim_data.X, sim_data.Bs, sim_data.B_bin, sim_data.coords, sim_data.mask
    print('--- Mask by neighbor constraint: \n {} \n ------ num: {}\n------ ratio: {}'.format(eud_mask, int(eud_mask.sum()), int(eud_mask.sum() / args.d_X ** 2)))

    X = check_tensor(X, dtype=torch.float32)
    X = X - X.mean(dim=0)
    T = check_tensor(torch.arange(X.shape[0]), dtype=torch.float32).reshape(-1, 1)
    coords = repeat(check_tensor(coords, dtype=torch.float32), 'j k -> i j k', i=X.shape[0])   
    Bs = check_tensor(Bs, dtype=torch.float32)
    dataset = TensorDataset(X, T, Bs, coords)
        
    Bs_gt = check_tensor(Bs, dtype=torch.float32)

    B_init, reg_mask = lin_reg_init(X, thres=args.reg_thres)
    B_init = check_tensor(B_init, dtype=torch.float32)
    reg_mask = bin_mat(B_init)
    print('--- Mask by regression: \n {} \n------ num: {} \n------ ratio: {}'.format(reg_mask, int(reg_mask.sum()), int(reg_mask.sum() / args.d_X ** 2)))
    print('------ GT:\n{}'.format(B_bin))

    mask = eud_mask # * reg_mask

    model = GolemModel(args, args.d_X, sim_data.coords, mask=mask, in_dim=1, equal_variances=True, seed=1,)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f'--- Load checkpoint from {args.checkpoint_path}')
    else:
        if args.regression_init:
            for epoch in tqdm(range(args.init_epoch)):
                for batch_idx, (batch_X, batch_T, _, _) in enumerate(data_loader):
                    optimizer.zero_grad()  
                    B = model.generate_B(batch_T)
                    loss = torch.pow(B_init - B, 2).sum()
                    loss.backward()
                    optimizer.step()

            #save_epoch_log(args, model, B_init, X, T, -1)
            print(f"--- Init F based on linear regression, ultimate loss: {loss.item()}")
        else:
            B_init = check_tensor(torch.randn(args.d_X, args.d_X), dtype=torch.float32)
        
    for epoch in range(args.epoch):
        model.train()
        for X_batch, T_batch, B_batch, coords in data_loader:
            optimizer.zero_grad()
            losses = model(X_batch, T_batch)
            losses['total_loss'].backward()
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
    
            optimizer.step()
            
        if epoch % 100 == 0:
            model.eval()
            Bs_pred = model.generate_B(T)
            Bs_pred = check_array(Bs_pred)
            Bs_gt = check_array(Bs_gt)

            Bs_pred[:, np.mean(np.abs(Bs_pred), axis=0) < args.graph_thres] = 0
            save_epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch}')
            save_DAG(args.num, save_epoch_dir, Bs_pred, Bs_gt, graph_thres=args.graph_thres, add_value=False)
            labels = [f'x{i}' for i in range(args.d_X)]
            make_dots(Bs_pred[0], labels, os.path.join(save_epoch_dir, 'B_pred_dot.png'))
            make_dots(Bs_gt[0], labels, os.path.join(save_epoch_dir, 'B_gt_dot.png'))
            tpr_result = 0
            shd_result = 0
            for i in range(args.num):
                result = count_graph_accuracy(bin_mat(Bs_gt[i]), bin_mat(Bs_pred[i]))
                tpr_result += result['tpr']
                shd_result += result['shd']
            tpr_result /= args.num
            shd_result /= args.num
            print(f'- Epoch {epoch} \n--- Loss: { {l: round(losses[l].item(), 2) for l in losses.keys()} }')
            print('--- Avergead TPR: {:.3f}, Averaged SHD: {:.3f}'.format(tpr_result, shd_result))
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}', 'checkpoint.pth'))
            
            # pred_B = model(T)
            # print(pred_B[0], dataset.B)
            # fig = plot_solutions([B_gt.T, B_est, W_est, M_gt.T, M_est], ['B_gt', 'B_est', 'W_est', 'M_gt', 'M_est'], add_value=True, logger=self.logger)
            # self.logger.experiment.log({"Fig": [wandb.Image(fig)]})