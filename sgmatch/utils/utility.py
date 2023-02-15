from typing import List

import torch

from .constants import CONVS, ACTIVATION_LAYERS
# from torch import cuda

class Namespace():
    def __init__(self, **kwargs):
        self.update_default_arguments()
        self.__dict__.update(kwargs)

    def update_default_args(self):
        self.update_simgnn_default_args()
        self.update_gmn_default_args()
        self.update_graphsim_default_args()
        self.update_isonet_default_args()
        self.update_neuromatch_default_args()

    def update_simgnn_default_args(self):
        self.__dict__.update(ntn_slices        = 16,
                             filters           = [64, 32, 16],
                             mlp_neurons       = [32,16,8,4],
                             hist_bins         = 16,
                             conv              = 'GCN',
                             activation        = 'tanh',
                             activation_slope  = None,
                             include_histogram = True)

    def update_gmn_default_args(self):
        self.__dict__.update(edge_feature_dim            = None,
                             enc_edge_hidden_sizes       = None,
                             message_net_init_scale      = 0.1,
                             node_update_type            = 'residual',
                             use_reverse_direction       = True,
                             reverse_dir_param_different = True,
                             attention_sim_metric        = 'euclidean',
                             layer_norm                  = False)

    def update_graphsim_default_args(self):
        pass

    def update_isonet_default_args(self):
        pass

    def update_neuromatch_default_args(self):
        self.__dict__.update(conv_type = 'SAGEConv',
                             dropout   = 0.0,
                             skip      = 'learnable')

class GraphPair(torch_geometric.data.Data):
    """ 
    :param: edge_index_1 : Edge Index of the First Graph
    :param: edge_index_2 : Edge Index of the Second Graph in the pair
    :param: x_1 : Feature Matrix of the First Graph in the Pair
    :param: x_2 : Feature Matrix of the Second Graph in the Pair
       
    :returns: torch_geometric.data.Data object which comprises two graphs
    """
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t, ged, norm_ged, graph_sim):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.ged = ged
        self.norm_ged = norm_ged
        self.graph_sim = graph_sim

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_1":
            return self.x1.size(0)
        elif key == "edge_index_2":
            return self.x2.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(x_s = {self.x_s.shape}, edge_index_s = {self.edge_index_s.shape}, '
                f'x_t = {self.x_t.shape}, edge_index_t = {self.edge_index_t.shape}, '
                f'graph_sim = {self.graph_sim.shape})')

def setup_linear_nn(input_dim: int, hidden_sizes: List[int]):
    r"""
    """
    mlp = torch.nn.ModuleList()
    _in = input_dim
    for i in range(len(hidden_sizes)):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        _in = _out
    
    return mlp

def setup_LRL_nn(input_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu"):
    r"""
    """
    # XXX: Better to leave this up to MLP class?
    mlp = []
    _in = input_dim
    _activation = ACTIVATION_LAYERS[activation]
    for i in range(len(hidden_sizes) - 1):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        mlp.append(_activation())
        _in = _out
    mlp.append(torch.nn.Linear(_in, hidden_sizes[-1]))
    mlp = torch.nn.Sequential(*mlp)
    
    return mlp

def setup_conv_layers(input_dim, conv_type, filters):
    r"""
    """
    convs = torch.nn.ModuleList()
    _conv = CONVS[conv_type]
    num_layers = len(filters)
    _in = input_dim
    for i in range(num_layers):
        _out = filters[i]
        convs.append(_conv(in_channels=_in, out_channels=_out))
        _in = _out

    return convs

# def cudavar(x):
#     return x.cuda() if cuda.is_available() else x