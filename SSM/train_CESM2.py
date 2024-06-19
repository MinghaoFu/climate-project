import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import sys
sys.path.append('..')
from LiLY.modules.CESM2 import CESM2ModularShiftsFixedB
from LiLY.tools.utils import load_yaml, setup_seed
from LiLY.modules import CESM2
from LiLY.datasets.sim_dataset import TimeVaryingDataset, FlexDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import Caulimate.Data.CESM2.dataset as CESM2_ds
from Caulimate.Data.CESM2.dataset import CESM2_grouped_dataset, downscale_dataset   
import os
import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning.loggers import WandbLogger
from Caulimate.Utils.Tools import lin_reg_init, check_tensor, get_free_gpu, makedir
from Caulimate.Utils.GraphUtils import eudistance_mask

if torch.cuda.is_available():   
    os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()
    print(f"--- Selected GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}")

DATA_DIR = '/l/users/minghao.fu/minghao.fu/dataset/CESM2' # you could modify it to your path
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')
SST_DATA_PATH = os.path.join(DATA_DIR, "CESM2_pacific_SST.pkl")
SPACE_INDEX_DATA_PATH = os.path.join(DATA_DIR, "CESM2_pacific.pkl")
GROUP_DATA_DIR = os.path.join(DATA_DIR, "group_region/")
XR_DATA_PATH = os.path.join(DATA_DIR, "CESM2_pacific_grouped_SST.nc")

SAVE_DIR = '/l/users/minghao.fu/minghao.fu/logs/ClimateModel/SSM/CESM2' # model and logs save dir
makedir(SAVE_DIR)

config = {
    "GRAPH_THRES": 0.01,
    "CHECKPOINT": "/home/minghao.fu/workspace/icml2024/scripts/climate/80m7ag95/checkpoints/epoch=443-step=41736.ckpt",
    "DATASET": "seed_69_fixed_B_modular_4_2_6",
    "LOAD_CHECKPOINT": False,
    "LOG": "/home/minghao.fu/workspace/icml2024/log",
    "LOG_NAME": "diag",
    "MAX_EUD": 40,
    "REG_THRES": 0.12,
    "MCC": {
        "CORR": "Pearson",
        "FREQ": 1.0
    },
    "PARALLEL": {
        "AREA_IDX": 0,
        "N_AREA": 10
    },

    "PROJ_NAME": "climate",
    "ROOT": "/home/minghao.fu/workspace/icml2024/LiLY/data",
    "SPLINE": {
        "BINS": 8,
        "BOUND": 5,
        "OBS_DIM": 1,
        "OBS_EMBED_DIM": 2,
        "ORDER": "linear"
    },
    "VAE": {
        "BETA": 0.002,
        "BIAS": False,
        "B_SPARSITY": 0.0001,
        "CPU": 8,
        "DEC": {
            "DIST": "gaussian",
            "HIDDEN_DIM": 128,
            "OBS_NOISE": False
        },
        "DYN_DIM": 1,
        "DYN_EMBED_DIM": 2,
        "ENC": {
            "HIDDEN_DIM": 128
        },
        "EPOCHS": 10000,
        "GAMMA": 0.02,
        "GPU": [0],
        "GRAD_CLIP": None,
        "INFER_MODE": "F",
        "INPUT_DIM": 6000,
        "LAG": 2,
        "LENGTH": 1,
        "LR": 0.001,
        "NCLASS": 12,
        "N_VAL_SAMPLES": 1024,
        "PIN": True,
        "SIMGA": 0.01,
        "TRAIN_BS": 64,
        "TRANS_PRIOR": "NP",
        "VAL_BS": 256
    }
}

def main(args):
    
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../LiLY/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = config #load_yaml(abs_file_path)


    pl.seed_everything(args.seed)
    
    log_dir = os.path.join(SAVE_DIR, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    wandb_logger = WandbLogger(project=cfg['PROJ_NAME'], name='CESM2' + datetime.now().strftime("%Y_%m_%d-%%M%S"))#, save_dir=log_dir)
    dataset = downscale_dataset(DOWNSCALE_PATH, DOWNSCALE_METADATA_PATH, ts_len=cfg['VAE']['LAG'] + cfg['VAE']['LENGTH'], n_domains=cfg['VAE']['NCLASS']) #CESM2_ds.CESM2_grouped_dataset

    cfg['VAE']['INPUT_DIM'] = dataset.d_X
    cfg['VAE']['NCLASS'] = dataset.n_domains
    
    print(yaml.dump(cfg, default_flow_style=False))

    eud_mask = eudistance_mask(dataset.coords, cfg['MAX_EUD'])
    B_init, reg_mask = lin_reg_init(dataset.X, thres=cfg['REG_THRES'], mask=eud_mask)
    mask = eud_mask * reg_mask
    print('Linear regression mask nonzero ratio: {}'.format(mask.sum() / mask.sum()))

    #B_init = check_tensor((dataset.data['xt'].reshape(-1, cfg['VAE']['INPUT_DIM']), 2), dtype=torch.float32)
    # num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    #train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    train_data = dataset

    print(train_data.__len__())

    #print(f"Total number of indices (data points) in the dataset: {data_}")

    data_point = train_data.__getitem__(400)

    print(f"Data at index {500}:")
    print(f"Time series data (xt): {data_point['xt'].shape}")
    print(f"Context or category (ct): {data_point['ct']}")
    print(f"Time or sequence index (ht): {data_point['ht']}")
    print(f"Spatial data or coordinates (st): {data_point['st'].shape}")


    train_loader = DataLoader(train_data,
                              batch_size=cfg['VAE']['TRAIN_BS'],
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)


    #data_iter = iter(train_loader)
    #batch = next(data_iter)

    # batch is a dictionary with keys 'xt', 'ct', 'ht', 'st'
    # print the shape of data for each key in the batch
    #for key in batch.keys():
    #    print(f"Shape of data for {key}: {batch[key].shape}")


    val_loader = train_loader
    # val_loader = DataLoader(val_data, 
    #                         batch_size=cfg['VAE']['VAL_BS'], 
    #                         pin_memory=cfg['VAE']['PIN'],
    #                         num_workers=cfg['VAE']['CPU'],
    #                         shuffle=False)

    if cfg['LOAD_CHECKPOINT']:
        model = CESM2ModularShiftsFixedB.load_from_checkpoint(checkpoint_path=cfg['CHECKPOINT'], # if save hyperparameter
                                                         strict=False
                            #                              input_dim=cfg['VAE']['INPUT_DIM'],
                            # length=cfg['VAE']['LENGTH'],
                            # obs_dim=cfg['SPLINE']['OBS_DIM'],
                            # dyn_dim=cfg['VAE']['DYN_DIM'],
                            # lag=cfg['VAE']['LAG'],
                            # nclass=cfg['VAE']['NCLASS'],
                            # hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                            # dyn_embedding_dim=cfg['VAE']['DYN_EMBED_DIM'],
                            # obs_embedding_dim=cfg['SPLINE']['OBS_EMBED_DIM'],
                            # trans_prior=cfg['VAE']['TRANS_PRIOR'],
                            # lr=cfg['VAE']['LR'],
                            # infer_mode=cfg['VAE']['INFER_MODE'],
                            # bound=cfg['SPLINE']['BOUND'],
                            # count_bins=cfg['SPLINE']['BINS'],
                            # order=cfg['SPLINE']['ORDER'],
                            # beta=cfg['VAE']['BETA'],
                            # gamma=cfg['VAE']['GAMMA'],
                            # sigma=cfg['VAE']['SIMGA'],
                            # B_sparsity=cfg['VAE']['B_SPARSITY'],
                            # decoder_dist=cfg['VAE']['DEC']['DIST'],
                            # correlation=cfg['MCC']['CORR']
                            )
    else:
        model = CESM2ModularShiftsFixedB(input_dim=cfg['VAE']['INPUT_DIM'],
                            length=cfg['VAE']['LENGTH'],
                            obs_dim=cfg['SPLINE']['OBS_DIM'],
                            dyn_dim=cfg['VAE']['DYN_DIM'],
                            lag=cfg['VAE']['LAG'],
                            nclass=cfg['VAE']['NCLASS'],
                            hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                            dyn_embedding_dim=cfg['VAE']['DYN_EMBED_DIM'],
                            obs_embedding_dim=cfg['SPLINE']['OBS_EMBED_DIM'],
                            trans_prior=cfg['VAE']['TRANS_PRIOR'],
                            lr=cfg['VAE']['LR'],
                            infer_mode=cfg['VAE']['INFER_MODE'],
                            bound=cfg['SPLINE']['BOUND'],
                            count_bins=cfg['SPLINE']['BINS'],
                            order=cfg['SPLINE']['ORDER'],
                            beta=cfg['VAE']['BETA'],
                            gamma=cfg['VAE']['GAMMA'],
                            sigma=cfg['VAE']['SIMGA'],
                            B_sparsity=cfg['VAE']['B_SPARSITY'],
                            decoder_dist=cfg['VAE']['DEC']['DIST'],
                            obs_noise=cfg['VAE']['DEC']['OBS_NOISE'],
                            correlation=cfg['MCC']['CORR'],
                            B_init=B_init,
                            mask=mask)

    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='train_elbo_loss', 
                                          save_top_k=1, 
                                          mode='min')

    early_stop_callback = EarlyStopping(monitor="train_elbo_loss", 
                                        min_delta=0.00, 
                                        patience=50, 
                                        verbose=False, 
                                        mode="min")

    trainer = pl.Trainer(default_root_dir=log_dir,
                         accelerator="auto",
                         devices=1,
                         logger=wandb_logger,
                         #val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         callbacks=[checkpoint_callback]
                         )

    # Train the model
    trainer.fit(model, train_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='CESM2'
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)
