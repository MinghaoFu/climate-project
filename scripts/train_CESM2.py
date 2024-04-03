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
from LiLY.datasets.sim_dataset import TimeVaryingDataset, FlexDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
from pytorch_lightning.loggers import WandbLogger

from Caulimate.Utils.Tools import linear_regression_initialize_tensor, check_array, check_tensor
from Caulimate import CESM2

SST_DATA_PATH = "/l/users/minghao.fu/CESM2/CESM2_pacific_SST.pkl"
SPACE_INDEX_DATA_PATH = "/l/users/minghao.fu/CESM2/CESM2_pacific.pkl"
GROUP_DATA_DIR = "/l/users/minghao.fu/dataset/CESM2/group_region/"
XR_DATA_PATH = "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific_grouped_SST.nc"

def main(args):
    
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../LiLY/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)


    pl.seed_everything(args.seed)
    
    log_dir = os.path.join('.', cfg['PROJ_NAME'], cfg['DATASET'] + cfg['LOG_NAME'] + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    wandb_logger = WandbLogger(project=cfg['PROJ_NAME'], name=cfg['DATASET'] + cfg['LOG_NAME'] + datetime.now().strftime("%Y_%m_%d-%%M%S"))#, save_dir=log_dir)

    #dataset = CESM2.dataset.CESM2_grouped_dataset(XR_DATA_PATH, cfg['PARALLEL']['N_AREA'], ts_len=cfg['VAE']['LAG'] + cfg['VAE']['LENGTH'], n_domains=cfg['VAE']['NCLASS'])
    #area_data = dataset[cfg['PARALLEL']['AREA_IDX']]
    dataset = CESM2.dataset.CESM2_entire_grouped_dataset(XR_DATA_PATH, ts_len=cfg['VAE']['LAG'] + cfg['VAE']['LENGTH'], n_domains=cfg['VAE']['NCLASS'])
    area_data = dataset
    cfg['VAE']['INPUT_DIM'] = area_data.d_X
    cfg['VAE']['NCLASS'] = area_data.n_domains
    
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")
    
    #B_init = linear_regression_initialize_tensor(area_data.data['xt'].reshape(-1, cfg['VAE']['INPUT_DIM']))
    B_init = check_tensor(np.load('./B_init.npy'))
    #np.save('./B_init.npy', check_array(B_init))
    
    # num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    #train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    train_data = area_data
    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True,
                              )

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
                            B_init=B_init)

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
