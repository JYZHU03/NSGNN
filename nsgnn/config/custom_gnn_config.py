from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    # Use self-supervised loss
    cfg.gnn.loss_self_head = None
    # The balance coefficient between self-supervised loss and prediction loss
    cfg.gnn.loss_self_coefficient = 0.5
