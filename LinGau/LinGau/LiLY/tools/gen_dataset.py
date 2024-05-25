import os
import glob
import tqdm
import torch
import scipy
import random
import ipdb as pdb
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from scipy.linalg import orth
from sklearn.preprocessing import scale
from utils import create_sparse_transitions, controlable_sparse_transitions

from Caulimate.data.synthetic_dataset.simulate_graph import simulate_time_varying_DAGs, simulate_random_dag, simulate_weight

from Caulimate import mask_tri, check_array, is_pseudo_invertible

VALIDATION_RATIO = 0.2
root_dir = '../data'
standard_scaler = preprocessing.StandardScaler()

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def noisecoupled_gaussian_ts():
    lags = 2
    Nlayer = 3
    length = 1
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "noisecoupled_gaussian_ts_2lag")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
        # Modulate the noise scale with averaged history
        y_t = y_t * np.mean(y_l, axis=1)
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def pnl_gaussian_ts():
    lags = 2
    Nlayer = 3
    length = 1
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "pnl_ts_2lag")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B) 

def pnl_change_gaussian_ts(NClass=5):
    lags = 2
    Nlayer = 3
    length = 1
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "pnl_change_%d"%NClass)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    os.makedirs(os.path.join(path, "target"), exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
    # Source or Target datasets
    for dataset_idx in range(2):
        # Domain-varying edges
        edge_pairs = [(1,2), (3,4)]
        if dataset_idx == 0:
            edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
        else:
            # Slight extrapolation
            edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

        yt = []; xt = []; ct = []
        yt_ns = []; xt_ns = []; ct_ns = []

        for j in range(NClass):
            ct.append(j * np.ones(batch_size))
            y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
            y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
            for p_idx, pair in enumerate(edge_pairs):
                transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
            for i in range(lags):
                yt.append(y_l[:,i,:])
            mixedDat = np.copy(y_l)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_l = np.copy(mixedDat)
            for i in range(lags):
                xt.append(x_l[:,i,:])
                
            # Mixing function
            for i in range(length):
                # Transition function
                y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
                for l in range(lags):
                    y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
                y_t = leaky_ReLU(y_t, negSlope)
                yt.append(y_t)
                # Mixing function
                mixedDat = np.copy(y_t)
                for l in range(Nlayer - 1):
                    mixedDat = leaky_ReLU(mixedDat, negSlope)
                    mixedDat = np.dot(mixedDat, mixingList[l])
                x_t = np.copy(mixedDat)
                xt.append(x_t)
                y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

            yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
            yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
            yt = []; xt = []; ct = []
            if dataset_idx == 0:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
            else:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)                

        yt_ns = np.vstack(yt_ns)
        xt_ns = np.vstack(xt_ns)
        ct_ns = np.vstack(ct_ns)


        if dataset_idx == 0:
            np.savez(os.path.join(path, "source", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns)
        else:
            np.savez(os.path.join(path, "target", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns)


def pnl_modular_gaussian_ts(NClass=5):
    lags = 2
    Nlayer = 3
    length = 1
    condList = []
    negSlope = 0.2
    dyn_latent_size = 8
    obs_latent_size = 1
    latent_size = dyn_latent_size + obs_latent_size
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4
    varyMean = True

    path = os.path.join(root_dir, "pnl_modular_%d"%NClass)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    os.makedirs(os.path.join(path, "target"), exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
        for i in range(dyn_latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(dyn_latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    # Source or Target datasets
    for dataset_idx in range(2):
        # Domain-varying edges
        edge_pairs = [(1,2), (3,4)]
        if dataset_idx == 0:
            edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
        else:
            # Slight extrapolation
            edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

        # get modulation parameters
        varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
        if varyMean:
            meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
        else:
            meanMat = np.zeros((NClass, obs_latent_size))

        yt = []; xt = []; ct = []
        yt_ns = []; xt_ns = []; ct_ns = []

        for j in range(NClass):
            ct.append(j * np.ones(batch_size))
            y_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
            y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
            # Change observation
            y_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
            y_o = (y_o - np.mean(y_o, axis=0 ,keepdims=True)) / np.std(y_o, axis=0 ,keepdims=True)
            y_o = np.multiply(y_o, varMat[j,:])
            y_o = np.add(y_o, meanMat[j,:])
            # Change dynamics
            for p_idx, pair in enumerate(edge_pairs):
                transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
            # Mixing lags
            mixedDat = np.concatenate((y_l,y_o[:,:lags]), axis=-1)
            for i in range(lags):
                yt.append(np.copy(mixedDat[:,i,:]))
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_l = np.copy(mixedDat)
            for i in range(lags):
                xt.append(x_l[:,i,:])
            # Mixing future
            for i in range(length):
                # Generate noise term first
                y_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
                # Transition function
                for l in range(lags):
                    y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
                y_t = leaky_ReLU(y_t, negSlope)
                # Mixing function
                mixedDat = np.concatenate((y_t,y_o[:,lags+i]), axis=-1)
                yt.append(np.copy(mixedDat))
                for l in range(Nlayer - 1):
                    mixedDat = leaky_ReLU(mixedDat, negSlope)
                    mixedDat = np.dot(mixedDat, mixingList[l])
                x_t = np.copy(mixedDat)
                xt.append(x_t)
                y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

            yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
            ####################################################################################
            B_scale = 1
            B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                                (B_scale * 0.5, B_scale * 2.0))
            seed = 1
            B_bin = simulate_random_dag(latent_size, 1, 'ER', seed)
            B_mat = simulate_weight(B_bin, B_ranges, seed)
            assert np.linalg.det(np.eye(latent_size) - B_mat) != 0 # invertible
            Bs = np.repeat(B_mat[np.newaxis, :, :], batch_size, axis=0)
            I_B_inv = np.linalg.inv(np.repeat(np.eye(latent_size)[np.newaxis, :, :], batch_size, axis=0) - Bs) # if invertible
            xt = np.matmul(I_B_inv[:, None, :, :], xt[:, :, :, None]).squeeze(3)
            bt = Bs
            ####################################################################################
            yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
            yt = []; xt = []; ct = []
            if dataset_idx == 0:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "source", "varMat"), varMat)
                np.save(os.path.join(path, "source", "meanMat"), meanMat)

            else:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "target", "varMat"), varMat)
                np.save(os.path.join(path, "target", "meanMat"), meanMat)                

        yt_ns = np.vstack(yt_ns)
        xt_ns = np.vstack(xt_ns)
        ct_ns = np.vstack(ct_ns)


        if dataset_idx == 0:
            np.savez(os.path.join(path, "source", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns)
        else:
            np.savez(os.path.join(path, "target", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns)
            


def linear_nonGaussian():
    lags = 2
    Nlayer = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 1000000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "linear_nongaussian")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile

    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    # Mixing function
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    # Transition function
    y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
    # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    for l in range(lags):
        y_t += np.dot(y_l[:,l,:], transitions[l])
    # Mixing function
    mixedDat = np.copy(y_t)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_t = np.copy(mixedDat)

    np.savez(os.path.join(path, "data"), 
            yt = y_l, 
            yt_ = y_t, 
            xt = x_l, 
            xt_= x_t)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def linear_nonGaussian_ts():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "linear_nongaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.dot(y_l[:,l,:], transitions[l])
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def nonlinear_Gaussian_ts():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])

    f2 = nn.LeakyReLU(0.2) # (1)3

    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0, noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
        y_t = leaky_ReLU(y_t, negSlope)
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

def nonlinear_Gaussian_ts_deprecated():
    lags = 2
    Nlayer = 3
    length = 10
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])

    f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size), nn.LeakyReLU(0.2))
    f2 = nn.Sequential(nn.Linear(latent_size, latent_size), nn.LeakyReLU(0.2))
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0,noise_scale).rsample((batch_size, latent_size))
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        # pdb.set_trace()
        '''
        y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
        y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
        mixedDat = torch.cat([y_l1, y_l2], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        mixedDat = torch.from_numpy(y_l)
        mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
        mixedDat = torch.add(f1(mixedDat.float()), y_t)
        '''
        mixedDat = y_l[:,0,:] + y_l[:,1,:]
        for l in range(lags-1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            # mixedDat = sigmoidAct(mixedDat)
            mixedDat = np.dot(mixedDat, transitions[l])
        '''
        # y_t = leaky_ReLU(mixedDat + y_t, negSlope)
        y_t = f2(mixedDat).detach().numpy() # PNL
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

def nonlinear_Gaussian_ts_deprecated():
    lags = 2
    Nlayer = 3
    length = 10
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])

    f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size), nn.LeakyReLU(0.2))
    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.normal.Normal(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        # pdb.set_trace()
        
        y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
        y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
        mixedDat = torch.cat([y_l1, y_l2], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        mixedDat = torch.from_numpy(y_l)
        mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
        mixedDat = f1(mixedDat.float()).detach().numpy()
        '''
        '''
        mixedDat = y_l[:,0,:] + y_l[:,1,:]
        for l in range(lags-1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            # mixedDat = sigmoidAct(mixedDat)
            mixedDat = np.dot(mixedDat, transitions[l])
        '''

        y_t = leaky_ReLU(mixedDat + y_t, negSlope)
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

def nonlinear_nonGaussian_ts():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_nongaussian_ts")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])

    # f1 = nn.Sequential(nn.Linear(2*latent_size, latent_size),
    #                    nn.LeakyReLU(0.2),
    #                    nn.Linear(latent_size, latent_size),
    #                    nn.LeakyReLU(0.2),
    #                    nn.Linear(latent_size, latent_size)) 
    # # f1.apply(weigth_init)
    f2 = nn.LeakyReLU(0.2) # (1)3

    # # Mixing function
    # for i in range(length):
    #     # Transition function
    #     y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size))
    #     # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
    #     # pdb.set_trace()
    #     '''
    #     y_l1 = torch.from_numpy(np.dot(y_l[:,0,:], transitions[0]))
    #     y_l2 = torch.from_numpy(np.dot(y_l[:,1,:], transitions[1]))
    #     mixedDat = torch.cat([y_l1, y_l2], dim=1)
    #     mixedDat = f1(mixedDat.float()).detach().numpy()
    #     '''
    #     mixedDat = torch.from_numpy(y_l)
    #     # mixedDat = torch.cat([mixedDat[:,0,:], mixedDat[:,1,:]], dim=1)
    #     mixedDat = 2 * mixedDat[:,0,:] + mixedDat[:,1,:]
    #     mixedDat = torch.add(mixedDat.float(), y_t)
    #     '''
    #     mixedDat = y_l[:,0,:] + y_l[:,1,:]
    #     for l in range(lags-1):
    #         mixedDat = leaky_ReLU(mixedDat, negSlope)
    #         # mixedDat = sigmoidAct(mixedDat)
    #         mixedDat = np.dot(mixedDat, transitions[l])
    #     '''
    #     # y_t = leaky_ReLU(mixedDat + y_t, negSlope)
    #     # y_t = f2(mixedDat).detach().numpy() # PNL
    #     y_t = mixedDat.detach().numpy()
    #     yt.append(y_t)
    #     # Mixing function
    #     mixedDat = np.copy(y_t)
    #     for l in range(Nlayer - 1):
    #         mixedDat = leaky_ReLU(mixedDat, negSlope)
    #         mixedDat = np.dot(mixedDat, mixingList[l])
    #     x_t = np.copy(mixedDat)
    #     xt.append(x_t)
    #     # pdb.set_trace()
    #     y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    # Mixing function
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.sin(np.dot(y_l[:,l,:], transitions[l]))
        y_t = leaky_ReLU(y_t, negSlope)
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

def nonlinear_ns():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4
    noise_scale = [0.05, 0.1, 0.15] # (v1)
    # noise_scale = [0.01, 0.1, 1]
    # noise_scale = [0.01, 0.05, 0.1] 

    path = os.path.join(root_dir, "nonlinear_ns")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []

    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
            
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = torch.distributions.laplace.Laplace(0,noise_scale[j]).rsample((batch_size, latent_size)).numpy()
            for l in range(lags):
                y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

def nonlinear_gau_ns():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 50000
    Niter4condThresh = 1e4
    noise_scale = [0.05, 0.1, 0.15] # (v1)
    # noise_scale = [0.01, 0.1, 1]
    # noise_scale = [0.01, 0.05, 0.1] 

    path = os.path.join(root_dir, "nonlinear_gau_ns")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []

    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
            
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = torch.distributions.normal.Normal(0,noise_scale[j]).rsample((batch_size, latent_size)).numpy()
            for l in range(lags):
                y_t += np.sin(np.dot(y_l[:,l,:], transitions[l]))
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

def nonlinear_gau_cins(Nclass=20):
    """
    Crucial difference is latents are conditionally independent
    """
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gau_cins_%d"%Nclass)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)
    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    modMat = np.random.uniform(0, 1, (latent_size, Nclass))
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
            # y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))
            y_t = np.multiply(y_t, modMat[:, j])

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def nonlinear_gau_cins_sparse():
    """
    Crucial difference is latents are conditionally independent
    """
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 20
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "nonlinear_gau_cins_sparse")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mask = controlable_sparse_transitions(latent_size, lags, sparsity=0.3)
    for l in range(lags):
        transitions[l] = transitions[l] * mask
        
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    modMat = np.random.uniform(0, 1, (latent_size, Nclass))
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
            # y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))
            y_t = np.multiply(y_t, modMat[:, j])

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)

def instan_temporal():
    lags = 1
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 50000
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "instan_temporal")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
        
    # Mixing function
    # Zt = f(Zt-1, et) + AZt
    for i in range(length):
        # Transition function
        y_t = torch.distributions.laplace.Laplace(0,noise_scale).rsample((batch_size, latent_size)).numpy()
        # y_t = (y_t - np.mean(y_t, axis=0 ,keepdims=True)) / np.std(y_t, axis=0 ,keepdims=True)
        for l in range(lags):
            y_t += np.dot(y_l[:,l,:], transitions[l])
            y_t = leaky_ReLU(y_t, negSlope) # f(Zt-1, et) with LeakyRelu as AVF
            y_t += np.dot(y_t, transitions[l])
        yt.append(y_t)
        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)

    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)


def case1_dependency():
    lags = 2
    Nlayer = 3
    length = 4
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "case1_dependency")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    # create DAG randomly
    import networkx as nx
    from random import randint, random
    def random_dag(nodes: int, edges: int):
        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        G = nx.DiGraph()
        for i in range(nodes):
            G.add_node(i)
        while edges > 0:
            a = randint(0, nodes-1)
            b = a
            while b == a:
                b = randint(0, nodes-1)
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                edges -= 1
            else:
                # we closed a loop!
                G.remove_edge(a, b)
        return G
    DAG = random_dag(latent_size, 40)
    dag = nx.to_numpy_array(DAG)

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []
    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
    
    # Initialize the dataset
    for i in range(lags):
        yt.append(y_l[:,i,:])
    mixedDat = np.copy(y_l)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    x_l = np.copy(mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    # Generate time series dataset
    for i in range(length):
        # Transition function
        # y_t = np.random.normal(0, 0.1, (batch_size, latent_size))
        y_t = np.random.laplace(0, 0.1, (batch_size, latent_size))

        for l in range(lags):
            # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
            y_t += np.dot(y_l[:,l,:], transitions[l])
        y_t = np.dot(y_t, np.ones((latent_size,latent_size))-dag)
        yt.append(y_t)

        # Mixing function
        mixedDat = np.copy(y_t)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_t = np.copy(mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)


def case2_nonstationary_causal():
    lags = 2
    Nlayer = 3
    length = 4
    Nclass = 20
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    batch_size = 7500
    Niter4condThresh = 1e4

    path = os.path.join(root_dir, "case2_nonstationary_causal")
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()
        
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []
    # Mixing function
    for j in range(Nclass):
        ct.append(j * np.ones(batch_size))

        masks = create_sparse_transitions(latent_size, lags, j)
        for l in range(lags):
            transitions[l] = transitions[l] * masks[l]

        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        
        # Initialize the dataset
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
        # Generate time series dataset
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, 0.1, (batch_size, latent_size))

            for l in range(lags):
                # y_t += np.tanh(np.dot(y_l[:,l,:], transitions[l]))
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)

            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)

            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]
        
        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []
        
        for l in range(lags):
            B = transitions[l]
            np.save(os.path.join(path, "W%d%d"%(j, lags-l)), B)

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)

    np.savez(os.path.join(path, "data"), 
            yt = yt_ns, 
            xt = xt_ns,
            ct = ct_ns)


def gen_da_data_ortho(Nsegment, varyMean=False, seed=1):
    """
    generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
    we generate mixing matrices using random orthonormal matrices
    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
          Specifically for leaky activation we also have:
            - negSlope: slope for x < 0 in leaky ReLU
            - Niter4condThresh: number of random matricies to generate to ensure well conditioned
    OUTPUT:
      - output is a dictionary with the following values:
        - sources: original non-stationary source
        - obs: mixed sources
        - labels: segment labels (indicating the non stationarity in the data)
    """
    path = os.path.join(root_dir, "da_gau_%d"%Nsegment)
    os.makedirs(path, exist_ok=True)
    Ncomp = 4
    Ncomp_s = 2
    Nlayer = 3
    NsegmentObs = 7500
    negSlope = 0.2
    NonLin = 'leaky'
    source = 'Gaussian'
    np.random.seed(seed)
    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = np.random.laplace(0, 1, (Nobs, Ncomp))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = np.random.normal(0, 1, (Nobs, Ncomp))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = np.random.uniform(0.01, 3, (Ncomp_s, Nsegment))

    if varyMean:
        meanMat = np.random.uniform(-3, 3, (Ncomp_s, Nsegment))
    else:
        meanMat = np.zeros((Ncomp_s, Nsegment))
    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        dat[segID, -Ncomp_s:] = np.multiply(dat[segID, -Ncomp_s:], modMat[:, seg])
        dat[segID, -Ncomp_s:] = np.add(dat[segID, -Ncomp_s:], meanMat[:, seg])
        labels[segID] = seg

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # generate mixing matrices:
    # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
    # (either additive or more general!)
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(Ncomp)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    np.savez(os.path.join(path, "data"), 
             y = dat, 
             x = mixedDat,
             c = labels)

# def tvcd_modular_gaussian_ts(NClass=5):
    
#     lags = 2
#     Nlayer = 3
#     length = 1
#     condList = []
#     negSlope = 0.2
#     dyn_latent_size = 8
#     obs_latent_size = 1
#     latent_size = dyn_latent_size + obs_latent_size
#     obs_size = 9
#     transitions = []
#     noise_scale = 0.1
#     batch_size = 40000
#     Niter4condThresh = 1e4
#     varyMean = True

#     type = 'fixed_B'
#     B_scale = 1
#     B_ranges = ((B_scale * -2.0, B_scale * -0.5),
#                          (B_scale * 0.5, B_scale * 2.0))
#     seed = 1
    
#     path = os.path.join(root_dir, "relu_%s_modular_%d_%d_%d"%(type, NClass, obs_size, latent_size))
#     if type == 'vary_B':
#         Bs, B_bin = create_time_varying_Bs(obs_size, batch_size, 0.08, seed)
#         for B in Bs:
#             assert np.linalg.det(np.eye(obs_size) - B) != 0
#     elif type == 'fixed_B':
#         B_bin = simulate_random_dag(obs_size, 1, 'ER', seed)
#         B = simulate_weight(B_bin, B_ranges, seed)
#         assert np.linalg.det(np.eye(obs_size) - B) != 0
#         Bs = np.repeat(B[np.newaxis, :, :], batch_size, axis=0)
#         # B = create_fixed_B(obs_size, 0.08, seed)   
#         # Bs = np.repeat(B[np.newaxis, :, :], batch_size, axis=0)
#     elif type == 'zero_B':
#         B_bin = np.zeros((obs_size, obs_size))
#         Bs = np.repeat(B_bin[np.newaxis, :, :], batch_size, axis=0)
#     else:
#         raise ValueError('Invalid type')
            
#     os.makedirs(path, exist_ok=True)
#     os.makedirs(os.path.join(path, "source"), exist_ok=True)
#     os.makedirs(os.path.join(path, "target"), exist_ok=True)

#     for i in range(int(Niter4condThresh)):
#         # A = np.random.uniform(0,1, (Ncomp, Ncomp))
#         A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
#         for i in range(dyn_latent_size):
#             A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
#         condList.append(np.linalg.cond(A))

#     condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
#     for l in range(lags):
#         B = generateUniformMat(dyn_latent_size, condThresh)
#         transitions.append(B)
#     transitions.reverse()

#     mixingList = []
#     for l in range(Nlayer - 1):
#         # generate causal matrix first:
#         A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
#         mixingList.append(A)
#     if latent_size != obs_size:
#         A = np.random.rand(latent_size, obs_size)
#         while np.linalg.matrix_rank(A) < min(latent_size, obs_size):
#             A = np.random.rand(latent_size, obs_size)
#         mixingList.append(A)

#     # Source or Target datasets
#     for dataset_idx in range(2):
#         # Domain-varying edges
#         edge_pairs = [(1,2), (3,4)]
#         if dataset_idx == 0:
#             edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
#         else:
#             # Slight extrapolation
#             edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

#         # get modulation parameters
#         varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
#         if varyMean:
#             meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
#         else:
#             meanMat = np.zeros((NClass, obs_latent_size))

#         yt = []; xt = []; ct = []; ht = []; bt = []
#         yt_ns = []; xt_ns = []; ct_ns = []; ht_ns = []; bt_ns = []
    
            
#         for j in range(NClass):
#             ct.append(j * np.ones(batch_size))
#             ht.append(np.arange(batch_size))
#             y_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
#             y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
#             # Change observation
#             y_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
#             y_o = (y_o - np.mean(y_o, axis=0 ,keepdims=True)) / np.std(y_o, axis=0 ,keepdims=True)
#             y_o = np.multiply(y_o, varMat[j,:])
#             y_o = np.add(y_o, meanMat[j,:])
#             # Change dynamics
#             for p_idx, pair in enumerate(edge_pairs):
#                 transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
#             # Mixing lags
#             mixedDat = np.concatenate((y_l,y_o[:,:lags]), axis=-1)
#             for i in range(lags):
#                 yt.append(np.copy(mixedDat[:,i,:]))
#             for l in range(Nlayer - 1):
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[l])
#             if latent_size != obs_size:
#                 mixedDat = leaky_ReLU(mixedDat, negSlope)
#                 mixedDat = np.dot(mixedDat, mixingList[-1])
#             x_l = leaky_ReLU(np.copy(mixedDat), negSlope)
#             for i in range(lags):
#                 # (I-B)^{-1}x_l
#                 Is = np.repeat(np.eye(obs_size)[np.newaxis, :, :], batch_size, axis=0)
#                 tmp = np.matmul(np.linalg.inv(Is - Bs), np.expand_dims(x_l[:,i,:], axis=2))
#                 x_l[:,i,:] = np.squeeze(tmp, axis=2)
#                 xt.append(x_l[:,i,:])
            
#             # Mixing future
#             for i in range(length):
#                 # Generate noise term first
#                 y_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
#                 # Transition function
#                 for l in range(lags):
#                     y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
#                 y_t = leaky_ReLU(y_t, negSlope)
#                 # Mixing function
#                 mixedDat = np.concatenate((y_t,y_o[:,lags+i]), axis=-1)
#                 yt.append(np.copy(mixedDat))
#                 for l in range(Nlayer - 1):
#                     mixedDat = leaky_ReLU(mixedDat, negSlope)
#                     mixedDat = np.dot(mixedDat, mixingList[l])
#                 if latent_size != obs_size:
#                     mixedDat = leaky_ReLU(mixedDat, negSlope)
#                     mixedDat = np.dot(mixedDat, mixingList[-1])
#                 x_t = leaky_ReLU(np.copy(mixedDat), negSlope)
#                 # (I-B)^{-1}x_t
#                 Is = np.repeat(np.eye(obs_size)[np.newaxis, :, :], batch_size, axis=0)
#                 x_t = np.matmul(np.linalg.inv(Is - Bs), np.expand_dims(x_t, axis=2))
#                 x_t = np.squeeze(x_t, axis=2)
                
#                 xt.append(x_t)
#                 y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

#             yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0); ht = np.array(ht).transpose(1,0); bt = np.array(Bs)
#             yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct); ht_ns.append(ht); bt_ns.append(bt)
#             yt = []; xt = []; ct = []; ht = []; bt = []
            
#             if dataset_idx == 0:
#                 for l in range(lags):
#                     B = transitions[l]
#                     np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
#                 np.save(os.path.join(path, "source", "varMat"), varMat)
#                 np.save(os.path.join(path, "source", "meanMat"), meanMat)

#             else:
#                 for l in range(lags):
#                     B = transitions[l]
#                     np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)
#                 np.save(os.path.join(path, "target", "varMat"), varMat)
#                 np.save(os.path.join(path, "target", "meanMat"), meanMat)                

#         yt_ns = np.vstack(yt_ns)
#         xt_ns = np.vstack(xt_ns)
#         ct_ns = np.vstack(ct_ns)
#         ht_ns = np.vstack(ht_ns)
#         bt_ns = np.vstack(bt_ns)
        
#         np.save(os.path.join(path, "Bs"), Bs)
#         if dataset_idx == 0:
#             np.savez(os.path.join(path, "source", "data"), 
#                     yt = yt_ns, 
#                     xt = xt_ns,
#                     ct = ct_ns)
#         else:
#             np.savez(os.path.join(path, "target", "data"), 
#                     yt = yt_ns, 
#                     xt = xt_ns,
#                     ct = ct_ns)

def fixed_B_modular_gaussian_ts(NClass=4):
    lags = 2
    Nlayer = 2
    length = 1
    condList = []
    negSlope = 0.2
    dyn_latent_size = 1
    obs_latent_size = 1
    latent_size = dyn_latent_size + obs_latent_size
    observed_size = 6
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4
    varyMean = True
    seed = 6
    sparse_gen = False

    dataset_name = "seed_{}_fixed_B_modular_{}_{}_{}".format(seed, NClass, latent_size, observed_size)
    if sparse_gen:
        dataset_name += '_sparse_gen'
    path = os.path.join(root_dir, dataset_name)
    print('--- Save to {}'.format(path))
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    os.makedirs(os.path.join(path, "target"), exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
        for i in range(dyn_latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(dyn_latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    if sparse_gen:
        assert latent_size == 2 and observed_size == 6
        l2x_edges = [(0,0), (0,1), (0,2), (0,3), (1,3), (1,4), (1,5)]
        A = ortho_group.rvs(observed_size)
        A = A[:latent_size, :observed_size] 
        mask = np.zeros(A.shape)
        for p_idx, pair in enumerate(l2x_edges):
            mask[pair[0], pair[1]] = 1
        A = A * mask
        assert is_pseudo_invertible(A)
        mixingList.append(A)
    else:
        for l in range(Nlayer - 1):
            # generate causal matrix first:
            A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
            mixingList.append(A)
        A = ortho_group.rvs(observed_size)
        A = A[:latent_size, :]
        mixingList.append(A)

    # Source or Target datasets
    for dataset_idx in range(2):
        # Domain-varying edges
        edge_pairs = [(1,2), (3,4)]
        if dyn_latent_size == 4:
            edge_pairs = [(0,1), (2,3)]
        elif dyn_latent_size == 1:
            edge_pairs = [(0,0)]
            
        if dataset_idx == 0:
            edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
        else:
            # Slight extrapolation
            edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

        # get modulation parameters
        varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
        if varyMean:
            meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
        else:
            meanMat = np.zeros((NClass, obs_latent_size))

        yt = []; xt = []; ct = []; ht = []; bt = []; vt = []
        yt_ns = []; xt_ns = []; ct_ns = []; ht_ns = []; bt_ns = []; vt_ns = []

        ##############################################################
        B_scale = 1
        B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                            (B_scale * 0.5, B_scale * 2.0))
        if sparse_gen:
            B_bin = np.zeros((observed_size, observed_size))
            for i in range(1, observed_size):
                B_bin[i, i - 1] = 1
            B_mat = simulate_weight(B_bin, B_ranges, seed)
        else:
            # B_bin = simulate_random_dag(observed_size, 1, 'ER', seed)
            B_bin = np.zeros((observed_size, observed_size))
            for i in range(1, observed_size):
                B_bin[i, i - 1] = 1
            B_mat = simulate_weight(B_bin, B_ranges, seed)
            assert np.linalg.det(np.eye(observed_size) - B_mat ) != 0 # invertible
        x_noise = np.random.normal(0, 0.1, (batch_size, lags + length, obs_latent_size))
        ##############################################################
        for j in range(NClass):
            
            ct.append(j * np.ones(batch_size))
            ht.append(np.arange(batch_size))
            y_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
            y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
            # Change observation
            y_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
            y_o = (y_o - np.mean(y_o, axis=0 ,keepdims=True)) / np.std(y_o, axis=0 ,keepdims=True)
            y_o = np.multiply(y_o, varMat[j,:])
            y_o = np.add(y_o, meanMat[j,:])
            # Change dynamics
            for p_idx, pair in enumerate(edge_pairs):
                transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
            # Mixing lags
            mixedDat = np.concatenate((y_l,y_o[:,:lags]), axis=-1)
            for i in range(lags):
                yt.append(np.copy(mixedDat[:,i,:]))
            for mixingMat in mixingList:
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingMat)
            #mixedDat = np.matmul(mixedDat, I_B_inv)
            x_l = np.copy(mixedDat)
            for i in range(lags):
                xt.append(x_l[:,i,:])
            # Mixing future
            for i in range(length):
                # Generate noise term first
                y_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
                # Transition function
                for l in range(lags):
                    y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
                y_t = leaky_ReLU(y_t, negSlope)
                # Mixing function
                mixedDat = np.concatenate((y_t,y_o[:,lags+i]), axis=-1)
                yt.append(np.copy(mixedDat))
                for mixingMat in mixingList:
                    mixedDat = leaky_ReLU(mixedDat, negSlope)
                    mixedDat = np.dot(mixedDat, mixingMat)
                #mixedDat += np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
                #mixedDat = np.squeeze(np.matmul(np.expand_dims(mixedDat, axis=1), I_B_inv), axis=1)
                x_t = np.copy(mixedDat)
                xt.append(x_t)
                y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

            yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0); ht = np.array(ht).transpose(1,0); 
            ####################################################################################
            vt_ns.append(xt)
            # Bs = np.repeat(B_mat[np.newaxis, :, :], batch_size, axis=0)
            # I_B = np.repeat(np.eye(latent_size)[np.newaxis, :, :], batch_size, axis=0) - Bs # if invertible
            I_B = np.eye(observed_size) - B_mat
            xt = xt + np.random.normal(0, 0.1, (batch_size, lags+length, observed_size)) #np.matmul(xt, np.linalg.inv(I_B))
            bt = np.repeat(B_mat[np.newaxis, :, :], batch_size, axis=0)
            ####################################################################################
            yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct); ht_ns.append(ht); bt_ns.append(bt)
            yt = []; xt = []; ct = []; ht=[]; bt=[]; vt=[]
            if dataset_idx == 0:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "source", "varMat"), varMat)
                np.save(os.path.join(path, "source", "meanMat"), meanMat)
            else:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "target", "varMat"), varMat)
                np.save(os.path.join(path, "target", "meanMat"), meanMat)                

        yt_ns = np.vstack(yt_ns)
        xt_ns = np.vstack(xt_ns)
        ct_ns = np.vstack(ct_ns)
        ht_ns = np.vstack(ht_ns)
        bt_ns = np.vstack(bt_ns)
        vt_ns = np.vstack(vt_ns)
        
        np.save(os.path.join(path, "Bs"), bt_ns)


        if dataset_idx == 0:
            np.savez(os.path.join(path, "source", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns,
                    vt = vt_ns)
        else:
            np.savez(os.path.join(path, "target", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns,
                    vt = vt_ns)
            
def fixed_B_sparse_gen_modular_gaussian_ts(NClass=4):
    lags = 2
    length = 1
    condList = []
    negSlope = 0.2
    dyn_latent_size = 1
    obs_latent_size = 1
    latent_size = dyn_latent_size + obs_latent_size
    observed_size = 5
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4
    varyMean = True
    seed = 69

    path = os.path.join(root_dir, "seed_{}_fixed_B_sparse_gen_modular_{}_{}_{}".format(seed, NClass, latent_size, latent_size))
    print('--- Save to {}'.format(path))
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    os.makedirs(os.path.join(path, "target"), exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
        for i in range(dyn_latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(dyn_latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    # generate causal matrix first:
    L2X = [(0,1), (3,4)]
    mixingMat = ortho_group.rvs(observed_size)


    # Source or Target datasets
    for dataset_idx in range(2):
        # Domain-varying edges
        edge_pairs = [(1,2), (3,4)]
        if dyn_latent_size == 4:
            edge_pairs = [(0,1), (2,3)]
            
        if dataset_idx == 0:
            edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
        else:
            # Slight extrapolation
            edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

        # get modulation parameters
        varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
        if varyMean:
            meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
        else:
            meanMat = np.zeros((NClass, obs_latent_size))

        yt = []; xt = []; ct = []; ht = []; bt = []; vt = []
        yt_ns = []; xt_ns = []; ct_ns = []; ht_ns = []; bt_ns = []; vt_ns = []

        ##############################################################
        B_scale = 0.3
        B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                            (B_scale * 0.5, B_scale * 2.0))
        
        B_bin = simulate_random_dag(latent_size, 1, 'ER', seed)
        B_mat = simulate_weight(B_bin, B_ranges, seed)
        assert np.linalg.det(np.eye(latent_size) - B_mat ) != 0 # invertible
        ##############################################################
        for j in range(NClass):
            
            ct.append(j * np.ones(batch_size))
            ht.append(np.arange(batch_size))
            y_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
            y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
            # Change observation
            y_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
            y_o = (y_o - np.mean(y_o, axis=0 ,keepdims=True)) / np.std(y_o, axis=0 ,keepdims=True)
            y_o = np.multiply(y_o, varMat[j,:])
            y_o = np.add(y_o, meanMat[j,:])
            # Change dynamics
            for p_idx, pair in enumerate(edge_pairs):
                transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
            # Mixing lags
            mixedDat = np.concatenate((y_l,y_o[:,:lags]), axis=-1)
            for i in range(lags):
                yt.append(np.copy(mixedDat[:,i,:]))
            # generation function
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingMat)
            #mixedDat = np.matmul(mixedDat, I_B_inv)
            x_l = np.copy(mixedDat)
            for i in range(lags):
                xt.append(x_l[:,i,:])
            # Mixing future
            for i in range(length):
                # Generate noise term first
                y_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
                # Transition function
                for l in range(lags):
                    y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
                y_t = leaky_ReLU(y_t, negSlope)
                # Mixing function
                mixedDat = np.concatenate((y_t,y_o[:,lags+i]), axis=-1)
                yt.append(np.copy(mixedDat))
                # generation function
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingMat)
                #mixedDat = np.squeeze(np.matmul(np.expand_dims(mixedDat, axis=1), I_B_inv), axis=1)
                x_t = np.copy(mixedDat)
                xt.append(x_t)
                y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

            yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0); ht = np.array(ht).transpose(1,0); 
            ####################################################################################
            vt_ns.append(xt)
            # Bs = np.repeat(B_mat[np.newaxis, :, :], batch_size, axis=0)
            # I_B = np.repeat(np.eye(latent_size)[np.newaxis, :, :], batch_size, axis=0) - Bs # if invertible
            I_B = np.eye(latent_size) - B_mat
            xt = np.matmul(xt, np.linalg.inv(I_B))
            bt = np.repeat(B_mat[np.newaxis, :, :], batch_size, axis=0)
            ####################################################################################
            yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct); ht_ns.append(ht); bt_ns.append(bt)
            yt = []; xt = []; ct = []; ht=[]; bt=[]; vt=[]
            if dataset_idx == 0:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "source", "varMat"), varMat)
                np.save(os.path.join(path, "source", "meanMat"), meanMat)

            else:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "target", "varMat"), varMat)
                np.save(os.path.join(path, "target", "meanMat"), meanMat)                

        yt_ns = np.vstack(yt_ns)
        xt_ns = np.vstack(xt_ns)
        ct_ns = np.vstack(ct_ns)
        ht_ns = np.vstack(ht_ns)
        bt_ns = np.vstack(bt_ns)
        vt_ns = np.vstack(vt_ns)
        
        np.save(os.path.join(path, "Bs"), bt_ns)


        if dataset_idx == 0:
            np.savez(os.path.join(path, "source", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns,
                    vt = vt_ns)
        else:
            np.savez(os.path.join(path, "target", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns,
                    vt = vt_ns)

            
def vary_B_modular_gaussian_ts(NClass=4):
    lags = 2
    Nlayer = 3
    length = 1
    condList = []
    negSlope = 0.2
    dyn_latent_size = 8
    obs_latent_size = 1
    latent_size = dyn_latent_size + obs_latent_size
    transitions = []
    noise_scale = 0.1
    batch_size = 40000
    Niter4condThresh = 1e4
    varyMean = True

    path = os.path.join(root_dir, "vary_B_modular_{}_{}_{}".format(NClass, latent_size, latent_size))
    print('--- Save to {}'.format(path))
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)
    os.makedirs(os.path.join(path, "target"), exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (dyn_latent_size, dyn_latent_size))  # - 1
        for i in range(dyn_latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(dyn_latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

    # Source or Target datasets
    for dataset_idx in range(2):
        # Domain-varying edges
        edge_pairs = [(1,2), (3,4)]
        if dataset_idx == 0:
            edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
        else:
            # Slight extrapolation
            edge_weights = np.random.uniform(-1.25, 1.25,(NClass, len(edge_pairs)))

        # get modulation parameters
        varMat = np.random.uniform(0.01, 1, (NClass, obs_latent_size))
        if varyMean:
            meanMat = np.random.uniform(-0.5, 0.5, (NClass, obs_latent_size))
        else:
            meanMat = np.zeros((NClass, obs_latent_size))

        yt = []; xt = []; vt = []; ct = []; ht = []; bt = []
        yt_ns = []; xt_ns = []; vt_ns = []; ct_ns = []; ht_ns = []; bt_ns = []

        for j in range(NClass):
            
            ct.append(j * np.ones(batch_size))
            ht.append(np.arange(batch_size))
            y_l = np.random.normal(0, 1, (batch_size, lags, dyn_latent_size))
            y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
            # Change observation
            y_o = np.random.normal(0, 1, (batch_size, lags+length, obs_latent_size))
            y_o = (y_o - np.mean(y_o, axis=0 ,keepdims=True)) / np.std(y_o, axis=0 ,keepdims=True)
            y_o = np.multiply(y_o, varMat[j,:])
            y_o = np.add(y_o, meanMat[j,:])
            # Change dynamics
            for p_idx, pair in enumerate(edge_pairs):
                transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
            # Mixing lags
            mixedDat = np.concatenate((y_l,y_o[:,:lags]), axis=-1)
            for i in range(lags):
                yt.append(np.copy(mixedDat[:,i,:]))
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            #mixedDat = np.matmul(mixedDat, I_B_inv)
            x_l = np.copy(mixedDat)
            for i in range(lags):
                xt.append(x_l[:,i,:])
            # Mixing future
            for i in range(length):
                # Generate noise term first
                y_t = np.random.normal(0, noise_scale, (batch_size, dyn_latent_size))
                # Transition function
                for l in range(lags):
                    y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
                y_t = leaky_ReLU(y_t, negSlope)
                # Mixing function
                mixedDat = np.concatenate((y_t,y_o[:,lags+i]), axis=-1)
                yt.append(np.copy(mixedDat))
                for l in range(Nlayer - 1):
                    mixedDat = leaky_ReLU(mixedDat, negSlope)
                    mixedDat = np.dot(mixedDat, mixingList[l])
                #mixedDat = np.squeeze(np.matmul(np.expand_dims(mixedDat, axis=1), I_B_inv), axis=1)
                x_t = np.copy(mixedDat)
                xt.append(x_t)
                y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

            
            yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0); ht = np.array(ht).transpose(1,0); 
            #########################################################################################################
            B_scale = 1
            B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                                (B_scale * 0.5, B_scale * 2.0))
            Bs, B_bin = create_time_varying_Bs(latent_size, batch_size, 0.4, B_ranges, j)
            Bs = check_array(mask_tri(Bs, 3))
            I_B_inv = np.linalg.inv(np.repeat(np.eye(latent_size)[np.newaxis, :, :], batch_size, axis=0) - Bs) # if invertible
            #xt = xt + np.random.normal(0, noise_scale, (batch_size, lags+length, latent_size))
            vt = xt.copy()
            xt = np.matmul(I_B_inv[:, None, :, :], xt[:, :, :, None]).squeeze(3)
            bt = Bs
            #########################################################################################################
            yt_ns.append(yt); xt_ns.append(xt); vt_ns.append(vt); ct_ns.append(ct); ht_ns.append(ht); bt_ns.append(bt)
            yt = []; xt = []; vt = []; ct = []; ht=[]; bt=[]
            if dataset_idx == 0:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "source", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "source", "varMat"), varMat)
                np.save(os.path.join(path, "source", "meanMat"), meanMat)

            else:
                for l in range(lags):
                    B = transitions[l]
                    np.save(os.path.join(path, "target", "W%d_%d"%(lags-l, j)), B)
                np.save(os.path.join(path, "target", "varMat"), varMat)
                np.save(os.path.join(path, "target", "meanMat"), meanMat)                

        yt_ns = np.vstack(yt_ns)
        xt_ns = np.vstack(xt_ns)
        vt_ns = np.vstack(vt_ns)
        ct_ns = np.vstack(ct_ns)
        ht_ns = np.vstack(ht_ns)
        bt_ns = np.vstack(bt_ns)
        
        np.save(os.path.join(path, "Bs"), bt_ns)


        if dataset_idx == 0:
            np.savez(os.path.join(path, "source", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    vt = vt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns)
        else:
            np.savez(os.path.join(path, "target", "data"), 
                    yt = yt_ns, 
                    xt = xt_ns,
                    vt = vt_ns,
                    ct = ct_ns,
                    ht = ht_ns,
                    bt = bt_ns)
        
            
if __name__ == "__main__":
    #noisecoupled_gaussian_ts()
    # pnl_change_gaussian_ts(NClass=5)
    #pnl_modular_gaussian_ts(NClass=4)
    fixed_B_modular_gaussian_ts(NClass=4)