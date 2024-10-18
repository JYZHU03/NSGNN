from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types/embedding dim to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types/embedding dim to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    cfg.dataset.train_dataset = 'dataset20'

    cfg.dataset.test_dataset = 'Texas'

