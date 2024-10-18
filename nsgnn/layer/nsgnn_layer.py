import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERY_SMALL_NUMBER = 1e-12
INF = 1e20

class NSGNNLayer(nn.Module):
    """
    GNN layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, num_heads, act='relu', dropout=0.0,
                 layer_norm=False, batch_norm=True):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.activation = register.act_dict[act]

        self.scalable_run = False

        # Local message-passing model.
        self.local_gnn_with_edge_attr = False
        if local_gnn_type == 'None':
            raise ValueError(f"GNN architeture must be chosen: {format}")

        elif local_gnn_type == "GCN":
            # MPNNs with edge attributes support.
            self.local_gnn_with_edge_attr = True
            self.local_model = pygnn.GCNConv(dim_h, dim_h)

        elif local_gnn_type == "SAGE":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.SAGEConv(dim_h, dim_h)

        elif local_gnn_type == "CUST":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GENConv(dim_h, dim_h)

        elif local_gnn_type == "GEN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GENConv(dim_h, dim_h)

        elif local_gnn_type == "TAGC":
            self.local_gnn_with_edge_attr = True
            self.local_model = pygnn.TAGConv(dim_h, dim_h, K=2)

        elif local_gnn_type == "GeneralConv":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GeneralConv(dim_h, dim_h)

        elif local_gnn_type == "FAConv":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.EGConv(dim_h, dim_h)

        elif local_gnn_type == "Arma":
            self.local_gnn_with_edge_attr = True
            self.local_model = pygnn.ARMAConv(dim_h, dim_h)

        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = True
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)

        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)

        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h, out_channels=dim_h // num_heads, heads=num_heads, edge_dim=dim_h)

        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for GNNs.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)

        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)



    def forward(self, batch):

        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local GNNs with edge attributes.
        self.local_model: pygnn.conv.MessagePassing  # Typing hint.
        if self.local_gnn_with_edge_attr:
            h_local = self.local_model(h, batch.edge_index, batch.edge_attr.to(torch.float))
        else:
            h_local = self.local_model(h, batch.edge_index)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)
        h_out_list.append(h_local)

        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
