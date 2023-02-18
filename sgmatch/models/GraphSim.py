from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch.nn import Linear, Dropout, Sequential
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch, unbatch, degree
from torch_scatter import scatter_mean, scatter_add

from ..utils.utility import setup_linear_nn, setup_conv_layers, setup_LRL_nn
from ..utils.constants import ACTIVATION_LAYERS, ACTIVATIONS

class GraphSim(nn.Module):
    r"""
    End to end implementation of GraphSim from the `"Learning-based Efficient Graph Similarity Computation via Multi-Scale
    Convolutional Set Matching" <https://arxiv.org/pdf/1809.04440.pdf>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any and update argument description

    Args:
        input_dim (int): Input dimension of node feature embedding vectors
        gnn (str): Number of filters per convolutional layer in the graph 
            convolutional encoder model. (default: :obj:`[64, 32, 16]`)
        gnn_filters ([int]): Number of hidden neurons in each linear layer of 
            MLP for reducing dimensionality of concatenated output of neural 
            tensor network and histogram features. Note that the final scoring 
            weight tensor of size :obj:`[mlp_neurons[-1], 1]` is kept separate
            from the MLP, therefore specifying only the hidden layer sizes will
            suffice. (default: :obj:`[32,16,8,4]`)
        conv_filters (int): Hyperparameter controlling the number of bins in the node 
            ordering histogram scheme. (default: :obj:`16`)
        mlp_neurons ([int]): Type of graph convolutional architecture to be used for encoding
            (:obj:`'GCN'` or :obj:`'SAGE'` or :obj:`'GAT'`) (default: :obj:`'GCN'`)
        padding_correction (bool): Type of activation used in Attention and NTN modules. 
            (:obj:`'sigmoid'` or :obj:`'relu'` or :obj:`'leaky_relu'` or :obj:`'tanh'`) 
            (default: :obj:`'tanh`)
        resize_dim (int): Slope of function for leaky_relu activation. 
            (default: :obj:`None`)
        resize_mode (str, optional):
        gnn_activation (str, optional): (default: :obj:`relu`)
        mlp_activation (str, optional): (default: :obj:`relu`)
        activation_slope (int, optional): (default: :obj:`0.1`)
    """
    def __init__(self, input_dim: int, gnn: str = "GCN", gnn_filters: List[int] = [64, 32, 16], conv_filters: Sequential = None, 
                 mlp_neurons: List[int] = [32,16,8,4,1], padding_correction: bool = True, resize_dim: int = 10, 
                 resize_mode = "bilinear", gnn_activation: str = "relu", mlp_activation: str = "relu", 
                 activation_slope: Optional[float] = 0.1):
        super(GraphSim, self).__init__()
        self.input_dim = input_dim
        self.gnn_type = gnn
        self.gnn_filters = gnn_filters
        self.gnn_activation = gnn_activation
        self.padding_correction = padding_correction
        self.sim_mat_dim = resize_dim
        self.resize_mode = resize_mode

        # Convolution Layer
        # XXX: Should users be allowed to pass torch.nn.Sequential layers for Conv directly?
        # XXX: Do we need to make additional Image Conv setup utility methods?
        self.conv_filters = conv_filters

        # MLP Layer which takes Convolution Output as Input
        self.mlp_neurons = mlp_neurons
        self.mlp_activation = mlp_activation

        self.setup_layers()
        # self.reset_parameters()

    def setup_layers(self):
        # GCN Layers 
        self.gnn_layers = setup_conv_layers(self.input_dim, self.gnn_type, filters=self.gnn_filters)

        # Fully Connected Layer - defined in the forward method
        W = torch.randn(2, 1, self.sim_mat_dim, self.sim_mat_dim) # Dummy Matrix
        W = self.conv_filters(W).view(2,-1)
        self.mlp = setup_LRL_nn(input_dim=W.shape[1], hidden_sizes=self.mlp_neurons, activation=self.mlp_activation)
        del W

    def reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameter()

        # TODO: Test correctness
        self.conv_filters.reset_parameters()
        self.mlp.reset_parameters()
        

    def forward(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, batch_i:Tensor, batch_j:Tensor):
        """
         Forward pass with graphs.
         :param x_i (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'i' Graphs Features.
         :param x_j (Tensor): A (N_1+N_2+...+N_B, D) tensor containing 'j' Graphs Features
         :param edge_index_i (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'i'
         :param edge_index_j (Tensor) : A (2, num_edges) tensor containing edges of Graphs in 'j'
         :param batch_i (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :param batch_j (Tensor) : A (B,) tensor containing information of the graph each node belongs to
         :return score (Tensor): Similarity score.
         """
        # Tensor of number of nodes in each graph
        N_i, N_j = degree(batch_i), degree(batch_j) # Size (B,)
        N_i_j = torch.maximum(N_i, N_j) # (B,)
        B = batch_i.shape[0]

        # Converting Input Nodes to Similarity Matrices
        sim_matrix_list = []
        gnn_activation = ACTIVATION_LAYERS[self.gnn_activation]
        for layer_num, gnn_layer in enumerate(self.gnn_layers):
            # Pass through GNN
            x_i = gnn_layer(x_i)
            x_j = gnn_layer(x_j)

            if layer_num != len(self.gnn_layers)-1:
                x_i = gnn_activation(x_i) # Default is a ReLU activation
                x_i = Dropout(x_i, p=self.p, training=self.training)
                x_j = gnn_activation(x_j)
                x_j = Dropout(x_j, p=self.p, training=self.training)

            # Generate Similarity Matrix after (layer_num + 1)th GNN Embedding Pass
            h_i, mask_i = to_dense_batch(x_i, batch_i) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded 
            h_j, mask_j = to_dense_batch(x_j, batch_j) # (B, N_max, D), {0,1}^(B, N_max) - 1 if true node, 0 if padded
            sim_matrix = torch.matmul(h_i, h_j.permute(0,2,1)) # (B, N_max_i, D) * (B, D, N_max_j) -> (B, N_max_i, N_max_j)

            # XXX: Can we just collect Similarity Matrices in this pass and perform other operations outside this loop?
            # Correcting Similarity Matrix Size as per Paper's Specifications
            if self.padding_correction:
                N_max_batch_i, N_max_batch_j = sim_matrix.shape[0], sim_matrix.shape[1] 
                pads_i, pads_j = N_i_j - N_max_batch_i, N_i_j - N_max_batch_j
                repadded_sim_matrices = list(map(lambda x, pad_i, pad_j: F.pad(x,(0,pad_i,0,pad_j)), 
                                                list(sim_matrix), pads_i, pads_j))
                resized_sim_matrices = list(map(lambda x: F.interpolate(x.unsqueeze(0), size=self.sim_mat_dim, mode=self.resize_mode), 
                                            repadded_sim_matrices))
                batched_resized_sim_matrices = torch.stack(resized_sim_matrices)
            else:
                batched_resized_sim_matrices = F.interpolate(sim_matrix, size=self.sim_mat_dim, mode=self.resize_mode)
            sim_matrix_list.append(batched_resized_sim_matrices) # [(B, N_reduced, N_reduced)]

        # Passing similarity images through Conv2d and MLP to get similarity score
        
        # sim_matrix_batch = torch.stack(sim_matrix_list, dim=-1) # (B, N_reduced, N_reduced, N_gnn_layers)
        # XXX: Can we use Group Convolutions instead of Looping over Convolved Multi-Scale Sim Matrices
        #sim_matrix_img_batch = sim_matrix_batch.permute(0,3,1,2)
        image_embedding_list = list(map(lambda x: self.conv_filters(x.unsqueeze(0)).squeeze(0), sim_matrix_list)) # [(C,H,W),]
        similarity_scores = torch.stack(image_embedding_list).view(B,-1) # (B, C*H*W)

        # Passing Input to MLP
        self.mlp = setup_LRL_nn(input_dim=similarity_scores.shape[1], hidden_sizes=self.mlp_neurons)
        similarity_scores = self.mlp(similarity_scores)
        similarity_scores = torch.nn.Sigmoid(similarity_scores)
        
        return similarity_scores.view(-1)