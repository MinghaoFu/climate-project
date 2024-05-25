import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import sys

sys.path.append('..')
import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from LiLY.datasets.sim_dataset import TimeVaryingDataset
from LiLY.modules.CESM2 import CESM2ModularShiftsFixedB
from LiLY.modules.metrics.correlation import correlation
from LiLY.tools.utils import load_yaml, setup_seed
from torchvision.utils import save_image, make_grid
from Caulimate.Utils.Tools import check_array, makedir, check_tensor, linear_regression_initialize,large_scale_linear_regression_initialize
from Caulimate.Utils.GraphUtils import threshold_till_dag
from Caulimate.Utils.Tools import whiten_data
from Caulimate.Data.CESM2 import dataset
from Caulimate.Utils.Metrics.correlation import correlation, align_two_latents, align_different_latents
from Caulimate.Utils.Visualization import plot_causal_graph, quick_map

SST_DATA_PATH = "/l/users/minghao.fu/CESM2/CESM2_pacific_SST.pkl"
SPACE_INDEX_DATA_PATH = "/l/users/minghao.fu/CESM2/CESM2_pacific.pkl"
GROUP_DATA_DIR = "/l/users/minghao.fu/dataset/CESM2/group_region/"
XR_DATA_PATH = "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific_grouped_SST.nc"
CKP_PATH = "/home/minghao.fu/workspace/climate/scripts/climate/f7cciuvm/checkpoints/epoch=145-step=13724.ckpt"
TEST_SIZE = 100
RESULTS_SAVE_DIR = ''
DAG_THRES = 0.1
NUM_AREA = 1
TIME_IDX = 0
PLOT_HEATMAP = False
PLOT_DAG = False
PLOT_TREATMENT_EFFECT = True


makedir('./CESM2/supper_B', remove_exist=True)
makedir('./CESM2/adj_mats/', remove_exist=True)
makedir('./CESM2/treatment_effect/', remove_exist=True)
cfg = load_yaml('../LiLY/configs/CESM2.yaml')

train_data = dataset.CESM2_grouped_dataset(XR_DATA_PATH, num_area=NUM_AREA)
CESM2_coords = np.zeros((train_data.n_groups, 2))
CESM2_adj_mats = []
CESM2_group_indices = np.zeros(train_data.n_groups)
CESM2_treatment_effects = np.zeros((TEST_SIZE, cfg['VAE']['DYN_DIM'] + cfg['SPLINE']['OBS_DIM'], train_data.n_groups))
CESM2_area_te_lst = []
zs_lst = []
group_accumulator = 0

for area_idx in range(NUM_AREA):
    area_train_data = train_data[area_idx]
    # fixed for a specific area
    cfg['VAE']['INPUT_DIM'] = area_train_data.d_X
    cfg['VAE']['NCLASS'] = area_train_data.n_domains

    if area_train_data.d_X > 1000:
        B_init = large_scale_linear_regression_initialize(area_train_data.data['xt'].reshape(-1, cfg['VAE']['INPUT_DIM']))
    else:
        B_init = check_tensor(linear_regression_initialize(area_train_data.data['xt'].reshape(-1, cfg['VAE']['INPUT_DIM'])), dtype=torch.float32)
    train_loader = DataLoader(area_train_data, batch_size=TEST_SIZE, shuffle=True, pin_memory=not torch.cuda.is_available())

    #model = CESM2ModularShiftsFixedB.load_from_checkpoint(map_location=torch.device('cpu'), checkpoint_path=CKP_PATH)
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
                                correlation=cfg['MCC']['CORR'],
                                B_init=B_init)
    model.eval()
    batch = next(iter(train_loader))
    batch_size = batch['xt'].shape[0]
    
    mus, logvars, x_recon, y_recon, zs, Bs, coords = model.forward(batch)
    zs, _, _, _ = whiten_data(check_array(zs[:, -1, :]))
    
    if area_idx == 0:
        zs_center = zs
    else:
        _, zs = align_two_latents(zs_center, zs)

    latent_size =  zs.shape[-1]
    mus = mus.view(batch_size, -1, latent_size)
    logvars = logvars.view(batch_size, -1, latent_size)
    y_recon = y_recon.view(batch_size, -1, cfg['VAE']['INPUT_DIM'])
    
    Bs = check_array(Bs)
    x_recon = check_array(x_recon)
    y_recon = check_array(y_recon)

    # Bs[Bs < threshold] = 0
    # supper_B = np.all(Bs > 0, axis=0)
    supper_B = check_array(B_init)
    supper_B[np.abs(B_init) < DAG_THRES] = 0    
    supper_B, _ = threshold_till_dag(supper_B)

    #plt.savefig(f'./files/supper_B/map.png')
    #xr_data.isel(time=0).plot(x='lon', y='lat', ax=ax, add_colorbar=False, transform=ccrs.PlateCarree())
    CESM2_adj_mats.append(supper_B)
    CESM2_coords[group_accumulator : group_accumulator + area_train_data.d_X] = check_array(area_train_data.coords)
    CESM2_group_indices[group_accumulator : group_accumulator + area_train_data.d_X] = check_array(area_train_data.group_index)
    # Treatment Effect
    zs = check_tensor(zs, dtype=torch.float32).clone()
    x_recon = model.net._decode(zs)
    z_effects = []
    for i in range(zs.shape[-1]):
        zi = zs.clone()
        zi[:, i] = 0
        zi_effect = x_recon - model.net._decode(zi)
        CESM2_treatment_effects[:, i, group_accumulator : group_accumulator + area_train_data.d_X] = check_array(zi_effect)
    group_accumulator += area_train_data.d_X

# zs_lst = align_different_latents(zs_lst)
# for zs in zs_lst:    
#     zs = check_tensor(zs).clone()
#     import pdb; pdb.set_trace() 
#     x_recon = model.net._decode(zs)
#     for i in range(zs.shape[-1]):
#         zi = zs.clone()
#         zi[:, :, i] = 0
#         zi_effect = x_recon - model.net._decode(zi)
#         #corr, sort_idx, sort_zi_effect = correlation(center, zi_effect[:, -1, :].T, method='Pearson')
#         CESM2_treatment_effects[:, i, group_accumulator : group_accumulator + area_train_data.d_X] = check_array(zi_effect)
    
#create treatment_effects xarray
norms = np.linalg.norm(CESM2_treatment_effects, axis=-1, keepdims=True)
normalized_effects = CESM2_treatment_effects# / norms

xr_te_lst = []
for z_idx in range(CESM2_treatment_effects.shape[1]):
    xr_te_i = train_data.xr_ds.copy().isel(time=TIME_IDX)
    for idx, agroup_index in enumerate(CESM2_group_indices):
        mask = train_data.xr_ds.group_index == agroup_index
        # Assign values based on the mask
        group_value = normalized_effects[TIME_IDX, z_idx, idx]  
        xr_te_i.values[mask] = normalized_effects[TIME_IDX, z_idx, idx]  
        
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree(-120)})
    #plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1, wspace=0.2, hspace=0.2)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    xr_te_i.plot(x='lon', y='lat', ax=ax, transform=ccrs.PlateCarree())
    fig.savefig(f'./CESM2/treatment_effect/z{z_idx}.png', dpi=500)
    # save z_i treatment effects
    xr_te_lst.append(xr_te_i)

te_save_path=f"/l/users/minghao.fu/dataset/CESM2/CESM2_treatment_effects.nc"
xr_te = xr.concat(xr_te_lst, dim='zs')
xr_te.to_netcdf(te_save_path)

np.save(f'./results/treatment_effects.npy', CESM2_treatment_effects)
np.save(f'./results/coords.npy', CESM2_coords)
for area_idx, mat in enumerate(CESM2_adj_mats):
    np.save(f'./results/adj_mats/adj_mats_{area_idx}.npy', mat)

for area_idx, mat in enumerate(CESM2_adj_mats):
    fig, ax = quick_map()
    extent = [CESM2_coords[:, 1].min() - 1, CESM2_coords[:, 1].max() + 1, CESM2_coords[:, 0].min() - 1, CESM2_coords[:, 0].max() + 1]
    #ax.set_extent(extent)
    print(CESM2_coords.shape)
    plot_causal_graph(mat, CESM2_coords, ax)
    plt.savefig(f'./CESM2/supper_B/map_{area_idx}.png')
