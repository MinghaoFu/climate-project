import sys
sys.path.append('..')
from climate.LiLY.modules.tv_golem import GolemModel
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
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Visualization import save_DAG, make_dots
from Caulimate.Utils.Tools import check_tensor, check_array, load_yaml, makedir, lin_reg_init, dict_to_class, save_log
from Caulimate.Utils.GraphMetric import count_graph_accuracy
from Caulimate.Utils.GraphUtils import bin_mat, eudistance_mask
from Caulimate.Data import WB

SST_DATA_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/temperature_850/*.nc'
V_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/v_component_of_wind/*.nc'
U_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/u_component_of_wind/*.nc'

args = {
    'dataset': 'synthetic',
    'vary_type': 'exp_trig',
    'noise_type': 'gaussian_ev',
    'load_data': True,
    'graph_type': 'ER',
    'num': 6000,
    'scale': 0.5,
    'pi': 10,
    'd_X': 5,
    'degree': 4,
    'cos_len': 1000,
    'max_eud': 50,
    'equal_variances': True,
    'train': True,
    'pretrain': False,
    'checkpoint_path': None,
    'init_training': False,
    'loss': {
        'likelihood': 1.0,
        'L1': 1e-2,
        'dag': 1e-2,
    },
    'tol': 0.0,
    'epoch': 10000,
    'init_epoch': 100,
    'batch_size': 10000,
    'seed': 2,
    'graph_thres': 0.3,
    'DAG': 0.8,
    'save_dir': './WB',
    'decay_type': 'step',
    'optimizer': 'ADAM',
    'weight_decay': 0.0,
    'lr': 0.0001,
    'gradient_noise': None,
    'step_size': 1000,
    'gamma': 0.5,
    'decay': [200, 400, 800, 1000],
    'betas': [0.9, 0.999],
    'epsilon': 0.00000001,
}

if __name__ == '__main__':
    args = dict_to_class(**args)
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    #wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))#, save_dir=log_dir)
    makedir(args.save_dir, remove_exist=True)
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WB_dataset = WB.WeatherBench_dataset(SST_DATA_PATH, U_PATH, V_PATH, args.save_dir, level=850, ts_len=3, n_domains=12, resample_size='1D')
    X = WB_dataset.data['x']
    args.num, args.d_X = X.shape
    np.save(os.path.join(args.save_dir, 'coords.npy'), WB_dataset.coords)
    coords, mask = check_tensor(WB_dataset.coords), check_tensor(WB_dataset.mask)
                
    X = check_tensor(X, dtype=torch.float32)
    X = X - X.mean(dim=0)
    T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
    tensor_dataset = TensorDataset(X, T)
    B_init, reg_mask = lin_reg_init(X, thres=args.reg_thres)
    B_init = check_tensor(B_init, dtype=torch.float32)
    reg_mask = bin_mat(B_init)
    print('--- Mask by regression: \n {} \n------ num: {} \n------ ratio: {}'.format(reg_mask, int(reg_mask.sum()), int(reg_mask.sum() / args.d_X ** 2)))

    model = GolemModel(args, args.d_X, coords, in_dim=1, equal_variances=True, mask=WB_dataset.mask, B_init=check_tensor(WB_dataset.wind_adj_mat), seed=1,)
     
    data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
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
        if args.init_training:
            B_init = check_tensor(model.B_init, dtype=torch.float32)
            for epoch in tqdm(range(args.init_epoch)):
                for batch_idx, batch_X, batch_T in enumerate(data_loader):
                    optimizer.zero_grad()  
                    B = model.generate_B(batch_T)
                    loss = torch.pow(B_init - B, 2).sum()
                    loss.backward()
                    optimizer.step()
                #save_epoch_log(args, model, B_init, X, T, -1)
                print(f"--- Init function parameter, ultimate loss: {loss.item()}")
            else:
                B_init = check_tensor(torch.randn(args.d_X, args.d_X), dtype=torch.float32)
        
    for epoch in range(args.epoch):
        model.train()
        for X_batch, T_batch in data_loader:
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
            #Bs_pred[:, np.mean(np.abs(Bs_pred), axis=0) < args.graph_thres] = 0
            save_epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch}')
            makedir(save_epoch_dir)
            log = f'- Epoch {epoch} \n--- Loss: { {l: round(losses[l].item(), 2) for l in losses.keys()} } \n'
            print(log)
            save_log(os.path.join(args.save_dir, 'log.txt'), log)
            np.save(os.path.join(save_epoch_dir, 'Bs_pred.npy'), Bs_pred)
            #save_DAG(args.num, save_epoch_dir, Bs_pred, Bs_gt, graph_thres=args.graph_thres, add_value=False)
            #labels = [f'x{i}' for i in range(args.d_X)]
            #make_dots(Bs_pred[0], labels, os.path.join(save_epoch_dir, 'B_pred_dot.png'))
            #make_dots(Bs_gt[0], labels, os.path.join(save_epoch_dir, 'B_gt_dot.png'))
            # tpr_result = 0
            # shd_result = 0
            # for i in range(args.num):
            #     result = count_graph_accuracy(bin_mat(Bs_gt[i]), bin_mat(Bs_pred[i]))
            #     tpr_result += result['tpr']
            #     shd_result += result['shd']
            # tpr_result /= args.num
            # shd_result /= args.num
            
            #print('--- Avergead TPR: {:.3f}, Averaged SHD: {:.3f}'.format(tpr_result, shd_result))
            torch.save(model.state_dict(), os.path.join(save_epoch_dir, 'checkpoint.pth'))
            
            # pred_B = model(T)
            # print(pred_B[0], dataset.B)
            # fig = plot_solutions([B_gt.T, B_est, W_est, M_gt.T, M_est], ['B_gt', 'B_est', 'W_est', 'M_gt', 'M_est'], add_value=True, logger=self.logger)
            # self.logger.experiment.log({"Fig": [wandb.Image(fig)]})