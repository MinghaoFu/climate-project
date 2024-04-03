import numpy as np
import itertools
import networkx as nx

from .tools import check_array

def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    B = check_array(B)
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres

def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)

    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def is_pseudo_invertible(matrix):
    """
    Check if a given n x m matrix is pseudo-invertible.

    Parameters:
    - matrix (np.ndarray): An n x m matrix.

    Returns:
    - bool: True if the matrix is pseudo-invertible, False otherwise.
    """
    # Compute the singular value decomposition (SVD) of the matrix
    U, S, Vh = np.linalg.svd(matrix)

    # A matrix is pseudo-invertible if it has no zero singular values
    # Since singular values are non-negative, we check if they are all non-zero
    return np.all(S > np.finfo(float).eps)

def bin_mat(B):
    '''
        Binarize a (batched) matrix
    '''
    return np.where(B != 0, 1, 0)

def is_markov_equivalent(B1, B2):
    '''
        Judge whether two matrices have the same v-structures. a->b<-c
        row -> col
    '''
    def find_v_structures(adj_matrix):
        G = nx.DiGraph(adj_matrix)
        v_structures = []

        for node in G.nodes():
            parents = list(G.predecessors(node))
            for pair in itertools.combinations(parents, 2):
                if not G.has_edge(pair[0], pair[1]) and not G.has_edge(pair[1], pair[0]):
                    v_structures.append((pair[0], node, pair[1]))

        return v_structures
    
    def find_skeleton(B, v_structures):
        inds = []
        for v in v_structures:
            inds.extend([(v[0], v[1]), (v[1], v[0]), (v[2], v[1]), (v[1], v[2])])
        inds = np.array(inds)
        B[inds[:, 0], inds[:, 1]] = 0
        
        return np.logical_or(B, B.T).astype(int)
    
    B1 = bin_mat(B1)
    B2 = bin_mat(B2)
    v_structures1 = find_v_structures(B1)
    v_structures2 = find_v_structures(B2)
    sk1 = find_skeleton(B1, v_structures1)
    sk2 = find_skeleton(B2, v_structures1)

    return set(v_structures1) == set(v_structures2) and sk1 == sk2
    
# count graph accuracy
def count_graph_accuracy(B_bin_true, B_bin_est, check_input=False):
    # TODO F1 score
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}

