# time varying, linear gaussian, causal sufficiency
from Caulimate.Data.SimLinGau import SimLinGau
from Caulimate.Graph.golem import time_vary_golem
from Caulimate.Utils.Tools import check_array, check_tensor, makedir, linear_regression_initialize, load_yaml, linear_regression_initialize

import torch
import torch.optim as optim
import os
import numpy as np
import pytorch_lightning as fpl
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader, random_split
from datetime import datetime
from tqdm import tqdm


if __name__ == '__main__':
    args = load_yaml(os.path.join('..', 'LiLY', 'configs', 'golem.yaml'))
    wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))#, save_dir=log_dir)
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == 'synthetic':
        dataset_name = f'golem_{args.graph_type}_{args.degree}_{args.noise_type}_{args.num}_{args.d_X}_{args.cos_len}'
        args.save_dir = os.path.join(args.save_dir, dataset_name)
        makedir(args.save_dir, remove_exist=True)
        lin_gau = SimLinGau(args.num, 
                            args.noise_type, 
                            args.d_X, 
                            args.degree, 
                            args.t_period, 
                            dataset_name, 
                            args.seed, 
                            graph_type=args.graph_type, 
                            vary_func=np.cos)

    B_init = linear_regression_initialize(lin_gau.X, args.distance)
    B_init = check_tensor(B_init, dtype=torch.float32)
        
    model = time_vary_golem(args.save_dir,
                            args.d_X, 
                            args.m_embed_dim, 
                            args.encoder_hid_dim, 
                            args.t_period_est, 
                            args.distance,
                            args.loss,
                            sparse_tol=args.tol,
                            lr=args.lr, 
                            seed=args.seed,
                            B_init=B_init)
                            
    checkpoint_callback = ModelCheckpoint(monitor='total_loss', 
                                          save_top_k=1, 
                                          mode='min')

    early_stop_callback = EarlyStopping(monitor="total_loss", 
                                        min_delta=0.00, 
                                        patience=50, 
                                        verbose=False, 
                                        mode="min")
    trainer = pl.Trainer(default_root_dir=args.save_dir, 
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        val_check_interval=1,
        max_epochs=args.epoch,
        callbacks=[checkpoint_callback]
        )
    train_loader = DataLoader(lin_gau.tensor_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = train_loader
    trainer.fit(model, train_loader, val_loader)
    # elif args.dataset == 'CESM': 
    #     X, Bs = CESM_dataset.load_data(args)
                
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