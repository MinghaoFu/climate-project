import numpy as np
import pandas as pd
import os
import warnings
import torch
import itertools
import pytest
import shutil
import matplotlib.pyplot as plt
import yaml
import xarray as xr
import random
import torch.nn as nn

from torchsummary import summary
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from torch.utils.data import Dataset
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
from scipy import linalg, stats
from tqdm import tqdm
from time import time

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_yaml(filename, type='class'):
    """
    Load and print YAML config files
    """
    with open(filename, 'r') as stream:
        file = yaml.safe_load(stream)
        if type == 'class':
            return dict_to_class(**file)
        elif type == 'dict':
            return file

def makedir(path, remove_exist=False):
    if remove_exist and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_tensor(data, dtype=None, astype=None, device=None):
    if not torch.is_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        elif isinstance(data, (list, tuple)):
            data = torch.tensor(np.array(data))
        elif isinstance(data, xr.DataArray):
            data = torch.from_numpy(data.fillna(0).to_numpy())
        else:
            raise ValueError("Unsupported data type. Please provide a list, NumPy array, or PyTorch tensor.")
    
    if astype is not None:
        return data.type_as(astype)
    
    if dtype is None:
        dtype = data.dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device, dtype=dtype)

def check_array(data, dtype=None):
    """
    Convert any input data to a NumPy array.

    Args:
        data: Input data of any type, including tensors.

    Returns:
        numpy.ndarray: NumPy array representation (copy) of the input data.
    """
    if isinstance(data, np.ndarray):
        data = data.copy()

    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    if isinstance(data, list) or isinstance(data, tuple):
        data = np.array(data)

    if isinstance(data, int) or isinstance(data, float):
        data = np.array([data])

    if isinstance(data, dict):
        data = np.array(list(data.values()))
    
    if dtype is not None:
        data = data.astype(dtype)

    return data


def center_and_norm(x, axis=0):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = check_array(x)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    
    return (x - means) / stds

def whiten_data(X):
    """_summary_

    Args:
        X (_type_): (n_samples, dim)

    Returns:
        _type_: _description_
    """
    X = check_array(X)
    # Remove the mean
    X_mean = X.mean(axis=0)
    X_demeaned = X - X_mean
    
    # Compute the covariance of the mean-removed data
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    
    # Eigenvalue decomposition of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    
    # Compute the whitening matrix
    whitening_matrix = np.dot(eigen_vectors, np.diag(1.0 / np.sqrt(eigen_values)))
    
    # Transform the data using the whitening matrix
    X_whitened = np.dot(X_demeaned, whitening_matrix)
    
    return X_whitened, X_mean, np.mean(X_whitened, axis=0), np.var(X_whitened, axis=0)

def dict_to_class(**dict):
    class _dict_to_class:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return _dict_to_class(**dict)

def linear_regression_initialize_tensor(x, n_iter=2000, tol=1e-2) -> torch.Tensor:
    x = check_tensor(x, dtype=torch.float32)
    n, dim = x.shape
    B_init = check_tensor(torch.zeros((dim, dim)))
    # Define the linear regression model
    model = nn.Linear(dim - 1, 1, bias=False)
    model = model.to(x.device)
    
    for i in tqdm(range(dim)):
        # Select the features for the regression problem
        x_features = torch.cat((x[:, :i], x[:, i + 1:]), dim=1)
        y_target = x[:, i:i + 1]
        
        # Use gradient descent to optimize the weights
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        
        # Train the model (Here we should have a loop for epochs, but for simplicity, we just do it once)
        optimizer.zero_grad()
        outputs = model(x_features)
        loss = criterion(outputs, y_target)
        loss.backward()
        optimizer.step()
        for _ in range(n_iter):
            optimizer.zero_grad()  # Clear gradients for the next train
            outputs = model(x_features)  # Forward pass: Compute predicted y by passing x to the model
            loss = criterion(outputs, y_target)  # Compute loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Update model parameters
            if loss < tol:
                break
                
        # Update the B_init matrix with the learned coefficients
        with torch.no_grad():
            B_init[i, :i] = model.weight[:, :i]
            B_init[i, i + 1:] = model.weight[:, i:]
    
    return B_init

def large_scale_linear_regression_initialize(x, max_iter=1000, ) -> np.array:
    x = check_array(x)
    model = SGDRegressor(penalty='l1', alpha=0.1, max_iter=max_iter, tol=1e-3)
    n_samples, dim = x.shape
    B_init = np.zeros((dim, dim))
    for i in tqdm(range(dim)):
        model.fit(np.concatenate((x[:, :i], x[:, i + 1:]), axis=1), x[:, i])
        B_init[i][:i] = model.coef_[:i]#np.pad(np.insert(model.coef_, min(i, distance), 0.), (start, d - end), 'constant')
        B_init[i][i + 1:] = model.coef_[i: ]

    return B_init

def linear_regression_initialize(x, distance=None, ) -> np.array:
    x = check_array(x)
    model = LinearRegression()
    n, d = x.shape
    B_init = np.zeros((d, d))
    if distance == None:
        distance = d - 1
    for i in range(d):
        start = max(i - distance, 0)
        end = min(i + distance + 1, d)

        model.fit(np.concatenate((x[:, start : i], x[:, i + 1 : end]), axis=1), x[:, i])
        B_init[i][start : i] = model.coef_[ : min(i, distance)]#np.pad(np.insert(model.coef_, min(i, distance), 0.), (start, d - end), 'constant')
        B_init[i][i + 1 : end] = model.coef_[min(i, distance) : ]
    
    return B_init

def model_info(model, input_shape):
    summary(model, input_shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
    
    if torch.cuda.is_available():
        input_data = check_tensor(torch.randn([1] + list(input_shape)))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        # Calculate the elapsed time
        elapsed_time = start.elapsed_time(end)
        print(f"--- Elapsed time: {elapsed_time} ms")
    else:
        print("--- CUDA is not available. Please run this on a CUDA-enabled environment.")
    
    return total_params, trainable_params, elapsed_time
    
        
# def plot_solution(B_true, B_est, M_true, M_est, save_name=None, add_value=False, logger=None):
#     """Checkpointing after the training ends.

#     Args:
#         B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
#         B_est (numpy.ndarray): [d, d] estimated weighted matrix.
#         B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
#         save_name (str or None): Filename to solve the plot. Set to None
#             to disable. Default: None.
#     """
#     # Define a function to add values to the plot
#     def add_values_to_plot(ax, matrix):
#         for (i, j), val in np.ndenumerate(matrix):
#             if np.abs(val) > 0.1:
#                 ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
            
#     fig, axes = plt.subplots(figsize=(10, 4), ncols=4)

#     # Plot ground truth
#     im = axes[0].imshow(B_true, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[0].set_title("B_gt", fontsize=13)
#     axes[0].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[0], B_true)

#     # Plot estimated solution
#     im = axes[1].imshow(B_est, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[1].set_title("B_est", fontsize=13)
#     axes[1].set_yticklabels([])    # Remove yticks
#     axes[1].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[1], B_est)

#     # Plot post-processed solution
#     im = axes[2].imshow(M_true, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[2].set_title("M_gt", fontsize=13)
#     axes[2].set_yticklabels([])    # Remove yticks
#     axes[2].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[2], M_true)

#     im = axes[3].imshow(M_est, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[3].set_title("M_est", fontsize=13)
#     axes[3].set_yticklabels([])    # Remove yticks
#     axes[3].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[3], M_est)
        
#     # Adjust space between subplots and add colorbar
#     fig.subplots_adjust(wspace=0.1)
#     im_ratio = 4 / 10
#     cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
#     cbar.ax.tick_params(labelsize=13)

#     # Save or display the figure
#     if save_name is not None:
#         fig.savefig(save_name, bbox_inches='tight')
#     else:
#         plt.show()

#     # Return the figure
#     return fig

def coord_to_index(coord, shape):
    """
    Convert coordinate to index.

    Args:
        coord (tuple): Coordinate.
        shape (tuple): Shape of the array.

    Returns:
        int: Index.
    """
    return np.ravel_multi_index(coord, shape)

def index_to_coord(index, shape):
    """
    Convert index to coordinate.

    Args:
        index (int): Index.
        shape (tuple): Shape of the array.

    Returns:
        tuple: Coordinate.
    """
    return np.unravel_index(index, shape)