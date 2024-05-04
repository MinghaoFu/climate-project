import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import leap
import numpy as np
import sys
sys.path.append('..')
import scipy
import random
import seaborn as sns
from tqdm import tqdm
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
from LiLY.modules.golemmodel import GolemModel
from LiLY.modules.metrics.correlation import correlation
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from Caulimate.Utils.Tools import check_array, makedir, load_yaml, check_tensor, linear_regression_initialize,large_scale_linear_regression_initialize
from Caulimate.Utils.GraphUtils import threshold_till_dag
from Caulimate.Utils.Tools import whiten_data
from Caulimate.Data.CESM2 import dataset
from Caulimate.Utils.metrics.correlation import correlation, align_two_latents, align_different_latents

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


makedir('./LinGau/supper_B')
makedir('./LinGau/adj_mats/')
makedir('./LinGau/treatment_effect/')
args = load_yaml('../LiLY/configs/golem.yaml')

train_data = dataset.CESM2_grouped_dataset(XR_DATA_PATH, num_area=NUM_AREA)[0]
CESM2_coords = np.zeros((train_data.n_groups, 2))
CESM2_adj_mats = []
CESM2_group_indices = np.zeros(train_data.n_groups)
CESM2_area_te_lst = []
group_accumulator = 0


model = GolemModel(args, args.d_X, dataset.coords, in_dim=1, equal_variances=True, seed=1,)
