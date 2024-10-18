import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


class KernelPENodeEncoder(torch.nn.Module):
    """Configurable Network Science Measure Positional Encoding node encoder.

    Measure encoder of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with Measure encoder.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """


    kernel_type = None

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in

        pecfg = getattr(cfg, f"Netsci_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based network measure embedding
        num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type
        n_layers = pecfg.layers  # Num. layers in network measure encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw network measure normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass network measure also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'Netsci.kernel.times' values")

        measure_enc = getattr(batch, pestat_var)
        if self.raw_norm:
            measure_enc = self.raw_norm(measure_enc)
        measure_enc = self.pe_encoder(measure_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final network measure encoder to input embedding
        batch.x = torch.cat((h, measure_enc), 1)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', measure_enc) #setattr(x, 'y', v) is equivalent to ``x.y = v''
        return batch


class NetworkScienceNodeEncoder(torch.nn.Module):
    """Configurable 46 network measure encoding node encoder.

    Measure encoder of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with measure encoder.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in

        pecfg = getattr(cfg, f"Netsci_{self.kernel_type}")
        dim_pe = pecfg.dim_pe
        num_metrics = len(pecfg.SelectedMetrics)
        model_type = pecfg.model.lower()  # Encoder NN model type
        n_layers = pecfg.layers  # Num. layers in network measure encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw network measure normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass network measure also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_metrics)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_metrics, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_metrics, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_metrics, dim_pe)

        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"Netsci_feature"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'Netsci.kernel.times' values")

        measure_enc = getattr(batch, pestat_var)

        if self.raw_norm:
            measure_enc = self.raw_norm(measure_enc)
        measure_enc = self.pe_encoder(measure_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final embedding of network measure to input embedding
        batch.x = torch.cat((h, measure_enc), 1)
        if self.pass_as_var: ##没跑
            setattr(batch, f'pe_{self.kernel_type}', measure_enc) #setattr(x, 'y', v) is equivalent to ``x.y = v''
        return batch


class Graph_GraphletsNodeEncoder(torch.nn.Module):
    """Configurable Graphlets Encoding node encoder.

    Network measure encoder of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with Network measure encoder.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in

        pecfg = getattr(cfg, f"Netsci_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based network measure encoder embedding
        num_metrics = len(pecfg.SelectedMetrics)
        model_type = pecfg.model.lower()  # Encoder NN model type for Network measure encoders
        n_layers = pecfg.layers  # Num. layers in network measure encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw network measure encoder normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass network measure encoder also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_metrics)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_metrics, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_metrics, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_metrics, dim_pe)

        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"Graphlets_feature"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'Netsci_{self.kernel_type}.enable' to "
                             f"True, and also set 'Netsci.kernel.times' values")

        measure_enc = getattr(batch, pestat_var)

        if self.raw_norm:
            measure_enc = self.raw_norm(measure_enc)
        measure_enc = self.pe_encoder(measure_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final embedding of network measure encoder to input embedding
        batch.x = torch.cat((h, measure_enc), 1)
        # Keep network measure encoder also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var: ##没跑
            setattr(batch, f'pe_{self.kernel_type}', measure_enc) #setattr(x, 'y', v) is equivalent to ``x.y = v''
        return batch


@register_node_encoder('RWSE')
class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'

@register_node_encoder('NetSci')
class NetSciNodeEncoder(NetworkScienceNodeEncoder):
    """Network Science Measures.
    """
    kernel_type = 'NetSci'

@register_node_encoder('Graphlets')
class GraphletsNodeEncoder(Graph_GraphletsNodeEncoder):
    """Graphlets statistics.
    """
    kernel_type = 'Graphlets'

