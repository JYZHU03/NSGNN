import torch
from torch_geometric.utils import ( to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes


def compute_Netsci_stats(data, pe_types, is_undirected, cfg):
    """Precompute/process positional encodings for the given graph.

    Supported network statistics to precompute, selected by `pe_types`:
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    ‘NetSci’： Some networks science measures.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination.
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify network measure types.
    for t in pe_types:
        if t not in ['RWSE', 'NetSci', 'Graphlets']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.Netsci_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Network Science measures
    if 'NetSci' in pe_types:
        metrics_list = cfg.Netsci_NetSci.SelectedMetrics
        # select specific measures
        data.Netsci_feature = data.Netsci_feature[:, metrics_list]

    if 'Graphlets' in pe_types:
        metrics_list = cfg.Netsci_Graphlets.SelectedMetrics

        # select specific orbits
        data.Graphlets_feature = data.Graphlets_feature[:, metrics_list]

    # Verify network measure types.
    node_loss_head = [cfg.gnn.loss_self_head]
    for t in node_loss_head:
        if t == 'RWSE':
            if hasattr(data, 'node_measures'):
                data.node_measures = torch.hstack([data.node_measures, data.pestat_RWSE])
            else:
                data.node_measures = data.pestat_RWSE
        if t == 'NetSci':
            if hasattr(data, 'node_measures'):
                data.node_measures = torch.hstack([data.node_measures, data.Netsci_feature])
            else:
                data.node_measures = data.Netsci_feature
        if t == 'Graphlets':
            if hasattr(data, 'node_measures'):
                data.node_measures = torch.hstack([data.node_measures, data.Graphlets_feature])
            else:
                data.node_measures = data.Graphlets_feature
        if hasattr(data, 'node_measures'):
            cfg.share.node_measures_dim = data.node_measures.shape[1]


    return data


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)

    return rw_landing
