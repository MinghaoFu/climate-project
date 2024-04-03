import torch
import torch.nn as nn

import numpy as np


from .tools import check_tensor, check_array    


    

def sample_n_different_integers(n, low, high, random_seed=None):
    # Create a random number generator with a specified random seed (or without)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Check if the interval contains enough unique integers
    if high - low < n:
        raise ValueError("Interval does not contain enough unique integers.")

    # Create an array of all integers in the interval
    all_integers = np.arange(low, high)

    # Shuffle the integers and take the first 'n' as the sample
    rng.shuffle(all_integers)
    sampled_integers = all_integers[:n]
    
    return sampled_integers

def top_k_abs_tensor(tensor, k):
    '''
        tensor: (d, d)
        k: int
    '''
    d = tensor.shape[0]
    abs_tensor = torch.abs(tensor)
    _, indices = torch.topk(abs_tensor.view(-1), k)
    
    flat_tensor = tensor.view(-1)
    flat_zero_tensor = torch.zeros_like(flat_tensor)
    flat_zero_tensor[indices] = flat_tensor[indices]
    
    zero_tensor = check_tensor(flat_zero_tensor.view(d, d))
    
    
    # batch_size, d, _ = tensor.shape
    # values, indices = torch.topk(tensor.view(batch_size, -1), k=k, dim=-1)
    # result = torch.zeros_like(tensor).view(batch_size, -1)
    # result.scatter_(1, indices, values)
    # result = result.view(batch_size, d, d)
    return zero_tensor

def random_zero_array(arr, zero_ratio, constraint=None):
    '''
        Randomly set some elements in an array to 0
    '''
    if constraint is None:
        original_shape = arr.shape
        arr = arr.flatten()
        inds = np.random.choice(np.arange(len(arr)), size=int(len(arr) * zero_ratio), replace=False)
        arr[inds] = 0
        result = arr.reshape(original_shape)
    return result

def keep_top_k_elements(A, k):
    batch_size = A.shape[0]
    # Step 1: Find the top k values and their indices in the flattened version of each matrix
    topk_values, topk_indices = torch.topk(A.view(batch_size, -1), k, dim=1)

    # Step 2: Create a new tensor of zeros with the same shape as the original batch of matrices
    result = torch.zeros_like(A)

    # Step 3: Use the indices to place the top-k values into the zero tensor
    # We must expand the topk_indices to be broadcastable to the shape we need
    topk_indices = topk_indices.unsqueeze(2).expand(-1, -1, A.size(2))
    # Use scatter_ to place topk_values in the result tensor at the correct indices
    result.view(batch_size, -1).scatter_(1, topk_indices.view(batch_size, -1), topk_values)

    return result

def mask_tri(B, distance):
    '''
        B: (batch_size, d, d)
    '''
    B = check_tensor(B)
    batch_size, d, _ = B.shape
    l_mask = d - distance - 1
    mask_upper = torch.triu(torch.zeros((d, d)), diagonal=1)
    mask_upper[:l_mask, -l_mask:] = torch.triu(torch.ones((l_mask, l_mask)), diagonal=0)

    mask_lower = torch.tril(torch.zeros((d, d)), diagonal=-1)
    mask_lower[-l_mask:, :l_mask] = torch.tril(torch.ones((l_mask, l_mask)), diagonal=0)
    mask = mask_upper + mask_lower
    mask = mask.expand(batch_size, d, d)
    B = B * check_tensor(1 - mask)
    return B


if __name__ == "__main__":
    batch_size, num_rows, num_cols = 10, 5, 5  # Example dimensions
    k = 3
    A = torch.randn(batch_size, num_rows, num_cols)
    A_topk = keep_top_k_elements(A, k)
    print(A_topk)
