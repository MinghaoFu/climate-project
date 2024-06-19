import sys
sys.path.append('..')

from LiLY.modules.tv_golem import GolemModel
import torch
import torch.nn as nn
import torch.optim as optim
import os, pwd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb
import pytorch_lightning as fpl
import wandb
from pytorch_lightning.loggers import WandbLogger
from einops import repeat

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from tqdm import tqdm
from einops import repeat

from Caulimate.Data.SimLinGau import LinGauSuff
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Visualization import save_DAG, make_dots
from Caulimate.Utils.Tools import check_tensor, check_array, load_yaml, makedir, lin_reg_init, dict_to_class, save_log, bin_mat, center_and_norm, get_free_gpu
from Caulimate.Utils.GraphMetric import count_graph_accuracy
from Caulimate.Data.CESM2.dataset import CESM2_grouped_dataset, downscale_dataset
from Caulimate.Utils.GraphUtils import eudistance_mask

DATA_DIR = '/l/users/minghao.fu/minghao.fu/dataset/CESM2' # you could modify it to your path
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')

SAVE_DIR = '/l/users/minghao.fu/minghao.fu/dataset/ClimateModel/LinGau/CESM2' # model and logs save dir
makedir(SAVE_DIR)

if torch.cuda.is_available():   
    os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
    print(f"--- Selected GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}")


args = {
    'data_path': "/l/users/minghao.fu/minghao.fu/dataset/CESM2/CESM2_pacific_grouped_SST.nc",
    'noise_type': 'gaussian_ev',
    'load_data': True,
    'graph_type': 'ER',
    'num': 6000,
    'scale': 0.5,
    'pi': 10,
    'd_X': None,
    'degree': 4,
    'cos_len': 1000,
    'max_eud': 50,
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
    'reg_thres': 0.05,
    'ddp': False,
    'pre_epoch': 0,
    'epoch': 10000,
    'init_epoch': 100,
    'batch_size': 64,
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
    'save_dir': "/l/users/minghao.fu/minghao.fu/logs/ClimateModel/LinGau/CESM2",

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
    dataset = downscale_dataset(path=DOWNSCALE_PATH, metadata_path=DOWNSCALE_METADATA_PATH)#CESM2_grouped_dataset(args.data_path, num_area=1)[0]
    args.d_X = dataset.d_X
    args.num = dataset.n_samples
    eud_mask = eudistance_mask(dataset.coords, args.max_eud)
    wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('--- Mask by neighbor constraint: \n------ num: {}\n------ ratio: {}'.format(int(eud_mask.sum()), eud_mask.sum() / args.d_X ** 2))

    # X = check_tensor(X, dtype=torch.float32)
    # X = X - X.mean(dim=0)
    # T = check_tensor(torch.arange(X.shape[0]), dtype=torch.float32).reshape(-1, 1)
    # coords = repeat(check_tensor(coords, dtype=torch.float32), 'j k -> i j k', i=X.shape[0])   
    # check_tensor(Bs, dtype=torch.float32)
    # dataset = TensorDataset(X, T, Bs, coords)

    B_init, reg_mask = lin_reg_init(center_and_norm(dataset.X), thres=args.reg_thres, mask=eud_mask)
    B_init = check_tensor(B_init, dtype=torch.float32)
    print('--- B_init: \n {}'.format(B_init))
    reg_mask = bin_mat(B_init)
    print('--- Mask by regression: \n------ num: {} \n------ ratio: {}'.format(int(reg_mask.sum()), reg_mask.sum() / args.d_X ** 2))

    mask = eud_mask * reg_mask
    print('--- Mask in training: \n------ num: {} \n------ ratio: {}'.format(int(mask.sum()), mask.sum() / args.d_X ** 2))
    _Bs_gt = check_tensor(mask, dtype=torch.float32)
    Bs_gt = repeat(_Bs_gt, 'j k -> i j k', i=dataset.n_samples)

    # save the mask and coordinates
    np.save(os.path.join(args.save_dir, 'reg_mask.npy'), reg_mask)
    np.save(os.path.join(args.save_dir, 'eud_mask.npy'), eud_mask)
    np.save(os.path.join(args.save_dir, 'coords.npy'), dataset.coords)

    model = GolemModel(args, args.d_X, dataset.coords, mask=mask, in_dim=1, equal_variances=True, seed=1, fast=True)
    tensor_dataset = TensorDataset(check_tensor(dataset.X), check_tensor(dataset.T))
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
        if args.regression_init:
            for epoch in tqdm(range(args.init_epoch)):
                for batch_idx, (batch_X, batch_T) in enumerate(data_loader):
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
        for batched_data in data_loader:
            X_batch, T_batch = batched_data
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
            Bs_pred = model.generate_B(check_tensor(dataset.T))
            Bs_pred = check_array(Bs_pred)
            Bs_gt = check_array(Bs_gt)

            Bs_pred[:, np.mean(np.abs(Bs_pred), axis=0) < args.graph_thres] = 0
            save_epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch}')
            makedir(save_epoch_dir)
            np.save(os.path.join(save_epoch_dir, 'B_pred.npy'), Bs_pred)
            torch.save(model.state_dict(), os.path.join(save_epoch_dir, 'checkpoint.pth'))
            save_DAG(args.num, save_epoch_dir, Bs_pred, Bs_gt, graph_thres=args.graph_thres, add_value=False)
            print(f'- Epoch {epoch} \n--- Loss: { {l: round(losses[l].item(), 2) for l in losses.keys()} }')
            #print('--- Avergead TPR: {:.3f}, Averaged SHD: {:.3f}'.format(tpr_result, shd_result))
            
            # labels = [f'x{i}' for i in range(args.d_X)]
            # make_dots(Bs_pred[0], labels, os.path.join(save_epoch_dir, 'B_pred_dot.png'))
            # make_dots(Bs_gt[0], labels, os.path.join(save_epoch_dir, 'B_gt_dot.png'))
            # tpr_result = 0
            # shd_result = 0
            # for i in range(args.num):
            #     result = count_graph_accuracy(bin_mat(Bs_gt[i]), bin_mat(Bs_pred[i]))
            #     tpr_result += result['tpr']
            #     shd_result += result['shd']
            # tpr_result /= args.num
            # shd_result /= args.num
            

            # pred_B = model(T)
            # print(pred_B[0], dataset.B)
            # fig = plot_solutions([B_gt.T, B_est, W_est, M_gt.T, M_est], ['B_gt', 'B_est', 'W_est', 'M_gt', 'M_est'], add_value=True, logger=self.logger)
            # self.logger.experiment.log({"Fig": [wandb.Image(fig)]})