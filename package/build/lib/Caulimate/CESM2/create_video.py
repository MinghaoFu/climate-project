import matplotlib.pyplot as plt
import imageio

import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pickle
import cv2
import os
import glob
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from Caulimate.tools import makedir
from Caulimate.visual import create_video_from_figs, figures_slider_display, call_ffmpeg_generate_video
import warnings
warnings.filterwarnings("ignore")

SST_DATA_PATH = "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific_SST.pkl"
SPACE_INDEX_DATA_PATH = "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific.pkl"
VIS_DIR = "/l/users/minghao.fu/dataset/CESM2/figures/"
VIDEO_PATH = "/l/users/minghao.fu/dataset/CESM2/pacific_video.mp4"

def save_visualize_CESM2_figs(sst_path, space_index_path, vis_dir):
    f = open(sst_path, 'rb')
    SST = pickle.load(f)
    f.close()
    # metadata file
    f = open(space_index_path, 'rb')
    coords = pickle.load(f).drop_dims('time')
    f.close()

    coords['space_index'].values[coords.nnaSST]

    makedir(vis_dir)

    plot_functions = [] 
    for i in tqdm(range(len(SST))):
        #visualize input data using metadata
        tSST = np.empty(coords.dims.get('space_index'))
        tSST[~coords.nnaSST]=np.nan
        tSST[coords.nnaSST] = SST[i]
        tst = coords.copy()
        tst['SST']=('space_index', tSST)
        tst.unstack('space_index').SST.plot(x='lon', y='lat')
        plot_functions.append(plt)
        plt.title('CESM2_pacific_SST at time step {}'.format(i))
        plt.savefig(os.path.join(vis_dir, str(i)+'.png'))
        plt.close()
        
    call_ffmpeg_generate_video(vis_dir, '', suffix='.png')


if __name__ == "__main__":
    save_visualize_CESM2_figs(SST_DATA_PATH, SPACE_INDEX_DATA_PATH, VIS_DIR)
    create_video_from_figs(VIS_DIR, VIDEO_PATH)