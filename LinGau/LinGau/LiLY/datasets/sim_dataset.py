import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import ipdb as pdb

class StationaryDataset(Dataset):
    
    def __init__(self, directory, transition="noisecoupled_gaussian_ts_2lag"):
        super().__init__()
        self.path = os.path.join(directory, transition, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt}
        return sample

class VaryBDataset(Dataset):
    def __init__(self, directory, transition="pnl_change_20", dataset="source"):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt", "vt", "ct", "ht", "bt"]:
            self.data[key] = self.npz[key]
        
        #self.data['ht'] = (self.data['ht']) / np.std(self.data['ht'], axis=0, keepdims=True)
        #self.data['ct'] = (self.data['ct'] - np.mean(self.data['ct'], axis=0, keepdims=True)) / np.std(self.data['ct'], axis=0, keepdims=True)

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        vt = torch.from_numpy(self.data["vt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        ht = torch.from_numpy(self.data["ht"][idx].astype('float32'))
        bt = torch.from_numpy(self.data["bt"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt, "vt": vt, "ct": ct, "ht": ht, "bt": bt}
        return sample


class FlexDataset(Dataset):
    def __init__(self, directory, transition="pnl_change_20", dataset="source"):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        self.possible_keys = ["yt", "xt", "vt", "ct", "ht", "bt"]
        for key in self.possible_keys:
            if key in self.npz:
                self.data[key] = self.npz[key]
        
        #self.data['ht'] = (self.data['ht']) / np.std(self.data['ht'], axis=0, keepdims=True)
        #self.data['ct'] = (self.data['ct'] - np.mean(self.data['ct'], axis=0, keepdims=True)) / np.std(self.data['ct'], axis=0, keepdims=True)

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        sample = {}
        for key in self.possible_keys:
            if key in self.data.keys():
                sample[key] = torch.from_numpy(self.data[key][idx].astype('float32'))
        # yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        # xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        # vt = torch.from_numpy(self.data["vt"][idx].astype('float32'))
        # ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        # ht = torch.from_numpy(self.data["ht"][idx].astype('float32'))
        # bt = torch.from_numpy(self.data["bt"][idx].astype('float32'))
        # sample = {"yt": yt, "xt": xt, "vt": vt, "ct": ct, "ht": ht, "bt": bt}
        return sample

class TimeVaryingDataset(Dataset):
    
    def __init__(self, directory, transition="pnl_change_20", dataset="source"):
        super().__init__()
        self.path = os.path.join(directory, transition, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt", "ct"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt, "ct": ct}
        return sample

class DANS(Dataset):
    def __init__(self, directory, dataset="da_10"):
        super().__init__()
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["y", "x", "c"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        y = torch.from_numpy(self.data["y"][idx].astype('float32'))
        x = torch.from_numpy(self.data["x"][idx].astype('float32'))
        c = torch.from_numpy(self.data["c"][idx, None].astype('float32'))
        sample = {"y": y, "x": x, "c": c}
        return sample