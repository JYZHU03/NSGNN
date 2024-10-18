import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERY_SMALL_NUMBER = 1e-12
degree_ratio = 1
sparsity_ratio = 1
smoothness_ratio = 1

@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))


    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


@register_head('inductive_NetSci_node')
class GNNInductiveNetSciNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNetSciNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg))

        num_layers = cfg.gnn.layers_post_mp

        if cfg.gnn.loss_self_head is not None:
            self.node_target_dim = cfg.share.node_measures_dim
            layer_config = new_layer_config(dim_in, 1, num_layers, has_act=False, has_bias=True, cfg=cfg)
            self.node_post_mps = nn.ModuleList([MLP(layer_config) for _ in range(self.node_target_dim)])


    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        node_feature_pre = None
        node_feature_true = None
        emb_visual = False

        if emb_visual:
            list_visual_data = []
            embeddings = batch.x
            list_visual_data.append(embeddings)
            list_visual_data.append(batch.y)
            list_visual_data.append(batch.node_measures)
            # list_visual_data.append(batch.Netsci_feature)

        if cfg.gnn.loss_self_head is not None:
            node_feature_pre = torch.hstack([m(batch.x) for m in self.node_post_mps])  # {Tensor:(1024, 51)}
            node_feature_true = batch.node_measures

        batch = self.layer_post_mp(batch)

        pred, label = self._apply_index(batch)
        if emb_visual:
            return pred, label, node_feature_pre, node_feature_true, list_visual_data
        else:
            return pred, label, node_feature_pre, node_feature_true