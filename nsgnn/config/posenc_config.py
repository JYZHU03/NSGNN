from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Network Measures Encoding class.
    cfg.Netsci_RWSE = CN()
    cfg.Netsci_NetSci = CN()
    cfg.Netsci_Graphlets = CN()

    for name in ['Netsci_RWSE']:
        pecfg = getattr(cfg, name)

        # Use extended network measure encodings
        pecfg.enable = False

        # Neural-net model type within the encoder:
        # 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Measure Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in Measure encoder model
        pecfg.layers = 3

        # Choice of normalization applied to raw encoder stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending encoder to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # arguments to NetSci.
    for name in ['Netsci_NetSci']:
        pecfg = getattr(cfg, name)

        # Use extended network measure encodings
        pecfg.enable = False

        # Neural-net model type within the encoder:
        # 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Measure Encoding embedding
        pecfg.dim_pe = 28


        # Selected metrics of Network Measures
        pecfg.SelectedMetrics = [0, 2, 4]

        # Number of Network Measures
        pecfg.number_metrics = len(pecfg.SelectedMetrics)

        # Number of layers to MLP
        pecfg.layers = 6

        # Choice of normalization applied to raw encoder stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending encoder to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False


    # Config for Graphlets for Measure Encoder that use it.

    for name in ['Netsci_Graphlets']:
        pecfg = getattr(cfg, name)

        # Use extended network measure encodings
        pecfg.enable = False

        # Neural-net model type within the encoder:
        # 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Measure Encoding embedding
        pecfg.dim_pe = 28

        # Selected metrics of Network Science
        pecfg.SelectedMetrics = [0, 2, 4]

        # Number of Network Science metrics
        pecfg.number_metrics = len(pecfg.SelectedMetrics)

        # Number of layers to MLP
        pecfg.layers = 6

        # Choice of normalization applied to raw encoder stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending encoder to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False


    for name in ['Netsci_RWSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based encoder specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''
