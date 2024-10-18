from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_nsgnn')
def set_cfg_gt(cfg):

    # Argument group
    cfg.nsgnn = CN()

    # Type of Graph Neural layer to use
    cfg.nsgnn.layer_type = 'TAGC'

    # Number of GNN layers in the model
    cfg.nsgnn.layers = 3

    # Number of attention heads in the GAT
    cfg.nsgnn.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.nsgnn.dim_hidden = 64

    # Dropout in feed-forward module.
    cfg.nsgnn.dropout = 0.0

    # Dropout in self-attention.
    cfg.nsgnn.attn_dropout = 0.0

    cfg.nsgnn.layer_norm = False

    cfg.nsgnn.batch_norm = True

    cfg.nsgnn.residual = True

