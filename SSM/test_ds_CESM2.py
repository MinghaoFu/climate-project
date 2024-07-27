import sys
sys.path.append('..')

from LiLY.modules.tv_golem import GolemModel
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
from einops import repeat
from LiLY.modules.CESM2 import CESM2ModularShiftsFixedB

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from tqdm import tqdm
from einops import repeat

from Caulimate.Data.SimLinGau import LinGauSuff
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Visualization import save_DAG, make_dots, plot_causal_graph, quick_map
from Caulimate.Utils.Tools import check_tensor, check_array, load_yaml, makedir, lin_reg_init, dict_to_class, save_log, bin_mat, center_and_norm, get_free_gpu
from Caulimate.Utils.GraphMetric import count_graph_accuracy
from Caulimate.Data.CESM2.dataset import CESM2_grouped_dataset, downscale_dataset
from Caulimate.Utils.GraphUtils import eudistance_mask, decycle_till_dag

DATASET_DIR = os.environ.get('DATASET_DIR')
MODEL_DIR = os.environ.get('MODEL_DIR')

DATA_DIR = os.path.join(DATASET_DIR, 'CESM2')
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')

SAVE_DIR = os.path.join(MODEL_DIR, 'ClimateModel/LinGau/CESM2')
makedir(SAVE_DIR)
save_test_dir = './downscale_CESM2_eud_mask'

CKPT_PATH="/home/minghao.fu/workspace/climate/SSM/climate/qrjc7max/checkpoints/epoch=2218-step=208586.ckpt"

if torch.cuda.is_available():   
    os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
    print(f"--- Selected GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}")


args = {
    'data_path': "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific_grouped_SST.nc",
    'noise_type': 'gaussian_ev',
    'load_data': True,
    'graph_type': 'ER',
    'num': 6000,
    'scale': 0.5,
    'pi': 10,
    'd_X': None,
    'degree': 4,
    'cos_len': 1000,
    'max_eud': 40,
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
    'save_dir': "/l/users/minghao.fu/logs/ClimateModel/LinGau/CESM2",

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

model = CESM2ModularShiftsFixedB.load_from_checkpoint(checkpoint_path=CKPT_PATH, strict=False)
model.eval()


