import numpy as np
import os
import pickle

DATA_DIR = '.' # you could custom it 
DOWNSCALE_PATH = os.path.join(DATA_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATA_DIR, 'downscaled_metadata.pkl')

data = np.loadtxt(DOWNSCALE_PATH) # (n_times, n_regions)
f = open(DOWNSCALE_METADATA_PATH, 'rb')
meta_data = pickle.load(f)
coords = np.column_stack((meta_data.lat[meta_data.nnaSST].values, meta_data.lon[meta_data.nnaSST].values)) # (n_regions, 2), element=(lat, lon)

print(data.shape, coords.shape)
        
