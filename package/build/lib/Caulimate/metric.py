import numpy as np
import scipy.stats as stats
from .tools import check_array

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort


def mean_correlation_coefficient(data1, data2):
    """
    Calculate the mean correlation coefficient between corresponding rows of two 2D arrays.
    The input arrays must have the same shape (batch, dim).

    Parameters:
    - data1: First array-like of shape (batch, dim).
    - data2: Second array-like of shape (batch, dim), correlated against data1.

    Returns:
    - mean_corr (float): The mean correlation coefficient across all rows.
    """
    # Ensure the input arrays have the same shape
    data1 = check_array(data1)
    data2 = check_array(data2)
    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have the same shape.")

    batch_size = data1.shape[0]
    
    # Center the data
    data1_centered = data1 - np.mean(data1, axis=1, keepdims=True)
    data2_centered = data2 - np.mean(data2, axis=1, keepdims=True)

    # Compute the covariance between the corresponding rows
    covariance = np.sum(data1_centered * data2_centered, axis=1)

    # Compute the standard deviations of the rows
    std_data1 = np.sqrt(np.sum(data1_centered ** 2, axis=1))
    std_data2 = np.sqrt(np.sum(data2_centered ** 2, axis=1))

    # Compute correlation coefficients for each row
    correlation_coeffs = covariance / (std_data1 * std_data2)

    # Compute the mean correlation coefficient
    mean_corr = np.mean(correlation_coeffs)

    return mean_corr

def rank_correlation_coefficient(data1, data2):
    """
    Calculate the mean Spearman's rank correlation coefficient between corresponding rows
    of two 2D arrays. The input arrays must have the same shape (batch, dim).

    Parameters:
    - data1: First array-like of shape (batch, dim).
    - data2: Second array-like of shape (batch, dim), correlated against data1.

    Returns:
    - mean_rank_corr (float): The mean Spearman's rank correlation coefficient across all rows.
    """
    data1 = check_array(data1)
    data2 = check_array(data2)
    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have the same shape.")

    rank_corrs = []

    for i in range(data1.shape[0]):
        # Calculate Spearman's rank correlation for each pair of rows
        corr, _ = scipy.stats.spearmanr(data1[i], data2[i])
        rank_corrs.append(corr)

    # Compute the mean of the correlation coefficients
    mean_rank_corr = np.mean(rank_corrs)

    return mean_rank_corr


if __name__ == "__main__":
    # Example usage:
    # Create random data tensors with the same shape (batch, dim)
    batch, dim = 100, 10
    data1 = torch.randn(batch, dim)
    data2 = torch.randn(batch, dim)

    # Calculate the mean correlation coefficient
    mean_corr_coef = mean_correlation_coefficient(data1, data2)
    print(mean_corr_coef)
