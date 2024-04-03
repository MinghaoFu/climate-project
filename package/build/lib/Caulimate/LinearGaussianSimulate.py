import numpy as np
import networkx as nx
import igraph as ig

def simulate_time_vary_dags(d, f, n, graph_type, degree, ranges, t_start=0, seed=42):
    """
    bijt = _bij + f(t)
    """
    B_bin = simulate_random_dag(d, degree, graph_type, seed) 
    B = simulate_weight(B_bin, ranges, seed)
    t_ids = np.linspace(t_start, t_start + n)
    t_factor = np.array([f(t_idx) for t_idx in t_ids])
    Bs = B + t_factor * B_bin
    return Bs, t_factor

def simulate_sparse_dag(d, p, seed, i_inv=True):
    def simulate_sparse_dag(d, p, seed):
        G = nx.gnp_random_graph(d, p, directed=True, seed=seed)
        # while not nx.is_directed_acyclic_graph(G):
        #     G = nx.gnp_random_graph(d, p, directed=True, seed=seed)
        B = nx.adjacency_matrix(G).toarray()
        B, _ = threshold_till_dag(B)
    
    B = simulate_sparse_dag(d, p, seed)
    if i_inv:
        while np.linalg.det(np.eye(d) - B) == 0:
            B = simulate_sparse_dag(d, p, seed)
    return B

def simulate_random_dag(d, degree, graph_type, seed):
    """Simulate random DAG.

    Args:
        d (int): Number of nodes.
        degree (int): Degree of graph.
        graph_type ('ER' or 'SF'): Type of graph.
        rs (numpy.random.RandomState): Random number generator.
            Default: np.random.RandomState(1).

    Returns:
        numpy.ndarray: [d, d] binary adjacency matrix of DAG.
    """
    rs = np.random.RandomState(seed)
    
    def _random_permutation(B_bin):
        # np.random.permutation permutes first axis only
        P = rs.permutation(np.eye(B_bin.shape[0]))
        return P.T @ B_bin @ P
    
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    if graph_type == 'ER':
        B_bin = simulate_er_dag(d, degree, rs)
    elif graph_type == 'SF':
        B_bin = simulate_sf_dag(d, degree)
    else:
        raise ValueError("Unknown graph type.")
    return _random_permutation(B_bin)

def simulate_weight(B_bin, B_ranges, seed):
    """Simulate the weights of B_bin.

    Args:
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
        B_ranges (tuple): Disjoint weight ranges.
        rs (numpy.random.RandomState): Random number generator.
            Default: np.random.RandomState(1).

    Returns:
        numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
    """
    rs = np.random.RandomState(seed)
    B = np.zeros(B_bin.shape)
    S = rs.randint(len(B_ranges), size=B.shape)  # Which range
    for i, (low, high) in enumerate(B_ranges):
        U = rs.uniform(low=low, high=high, size=B.shape)
        B += B_bin * (S == i) * U
    return B
def multi_linear_sem(B, noise_type, rs=np.random.RandomState(1)):
    assert len(B.shape) == 3
    X = np.empty([B.shape[0], B.shape[1]])
    for idx, Bi in enumerate(B):
        X[idx:idx+1] = linear_sem(Bi, 1, noise_type, rs)

    return X

def linear_sem(B, n, noise_type, rs=np.random.RandomState(1)):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """
        def _simulate_single_equation(X, B_i):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(scale=1.0, size=n)
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=2.0)
                N_i = rs.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0, size=n)
            else:
                raise ValueError("Unknown noise type.")
            
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])
            
        return X