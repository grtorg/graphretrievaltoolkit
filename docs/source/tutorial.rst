Tutorial
=========================

SimGNN to Predict Graph Similarity on AIDS700nef Dataset
----------------------------------------------------------


Import Basic Libraries
^^^^^^^^^^^^^^^^^^^^^^^^
We first import all basic libraries before we begin::

    import tqdm
    import os
    import os.path as osp
    import torch
    from torch_geometric.loader import DataLoader
    import numpy as np


Now that we have the basic libraries in place, we import specific modules from :obj:`sgmatch` directory::

    from sgmatch.utils.utility import Namespace, GraphPair
    from sgmatch.models.matcher import graphMatcher


The :class:`sgmatch.utils.utility.Namespace` class serves as a container for hyperparameters and other parameters
used to instantiate the models.
The :class:`sgmatch.utils.utility.GraphPair` class is the fundamental building block of this toolkit. It builds on top of
:class:`PyTorch Geometric`'s API and creates a class which can be instantiated to return a graph pair or a batch of
graph pairs when invoked.
The :class:`sgmatch.models.matcher.graphMatcher` is a wrapper class for all the Graph Similarity / Graph Retrieval models
implemented in :obj:`Graph Retrieval Toolkit`.


Load and Create Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^
This tutorial involves prediction of Graph Similarity of the Graphs present in the 'AIDS700nef' dataset.
We first download the dataset from :class:`torch_geometric` as follows::

    from torch_geometric.datasets import GEDDataset
    name = "AIDS700nef"
    ROOT_DIR = '../'
    train_graphs = GEDDataset(root=os.path.join(ROOT_DIR, f'data/{name}/train'), train = True, name=name)
    test_graphs = GEDDataset(root=os.path.join(ROOT_DIR, f'data/{name}/test'), train = False, name=name)

    print(f"Number of Graphs in Train Set : {len(train_graphs)}")
    print(f"Number of Graphs in Test Set : {len(test_graphs)}")


Although we have downloaded the graphs from the AIDS700nef dataset, we are not done yet. Our objective
is to compute the Graph Similarity between two graphs and thus our model SimGNN takes as input a pair of graphs::

    ## *** Training Set Pair ***
    train_graph_pair_list = []
    # Making the Pairs of Graphs
    for graph_s_num, graph_s in enumerate(train_graphs):
        for graph_t in train_graphs:
            edge_index_s = graph_s.edge_index
            x_s = graph_s.x
            edge_index_t = graph_t.edge_index
            x_t = graph_t.x
            ged = train_graphs.ged[graph_s.i, graph_t.i]
            norm_ged = train_graphs.norm_ged[graph_s.i, graph_t.i]
            graph_sim = torch.exp(-norm_ged)
            
            # Making Graph Pair
            graph_pair = GraphPair(edge_index_s=edge_index_s, x_s=x_s, 
                                        edge_index_t=edge_index_t, x_t=x_t,
                                        ged=ged ,norm_ged=norm_ged, graph_sim = graph_sim)
            
            train_graph_pair_list.append(graph_pair)

    ## *** Test Set Pair ***
    test_graph_pair_list = []
    # Making the Pairs of Graphs
    for graph_s_num, graph_s in enumerate(test_graphs):
        for graph_t in train_graphs:
            edge_index_s = graph_s.edge_index
            x_s = graph_s.x
            edge_index_t = graph_t.edge_index
            x_t = graph_t.x
            ged = train_graphs.ged[graph_s.i, graph_t.i] # Yes, train_graphs.ged is correct
            norm_ged = train_graphs.norm_ged[graph_s.i, graph_t.i] # Yes, train_graphs.norm_ged is correct
            graph_sim = torch.exp(-norm_ged)
            
            # Making Graph Pair
            graph_pair = GraphPair(edge_index_s=edge_index_s, x_s=x_s, 
                                        edge_index_t=edge_index_t, x_t=x_t,
                                        ged=ged ,norm_ged=norm_ged, graph_sim = graph_sim)
            
            test_graph_pair_list.append(graph_pair)


For some usecases and to prevent model overfitting, we might also need to make a validation set. Although
dataset does not come built in with a validation set, we can create our own validation graph pair set as shown::

    val_idxs = np.random.randint(len(train_graph_pair_list), size=len(test_graph_pair_list))
    val_graph_pair_list = [train_graph_pair_list[idx] for idx in val_idxs]
    train_idxs = set(range(len(train_graph_pair_list))) - set(val_idxs)
    train_graph_pair_list = [train_graph_pair_list[idx] for idx in train_idxs]
    del val_idxs, train_idxs

    print("Number of Training Graph Pairs = {}".format(len(train_graph_pair_list)))
    print("Number of Validation Graph Pairs = {}".format(len(val_graph_pair_list)))
    print("Number of Test Graph Pairs = {}".format(len(test_graph_pair_list)))


Now that we have Training, Validation and Testing Graph Pair Data, we can create our own DataLoaders from this data::

    from torch_geometric.loader import DataLoader
    batch_size = 128
    train_loader = DataLoader(train_graph_pair_list, batch_size=batch_size, follow_batch=["x_s", "x_t"], shuffle=True)
    val_loader = DataLoader(val_graph_pair_list, batch_size=batch_size, follow_batch=["x_s", "x_t"], shuffle=True)
    test_loader = DataLoader(test_graph_pair_list, batch_size=batch_size, follow_batch=["x_s", "x_t"], shuffle=True)


Training the Model
^^^^^^^^^^^^^^^^^^^^

Now, we define a :class:`sgmatch.utils.utility.Namespace` object which sends arguments to the 
base :class:`sgmatch.models.matcher.graphMatcher` class to initialize our graph similarity model::

    av = Namespace(model_name        = "simgnn", 
                   ntn_slices        = 16,
                   filters           = [64, 32, 16],
                   mlp_neurons       = [32,16,8,4],
                   hist_bins         = 16,
                   conv              = 'GCN',
                   activation        = 'tanh',
                   activation_slope  = None,
                   include_histogram = True,
                   input_dim         = train_graphs.num_features)


We also define a training function which takes in train and validation graph pairs and train our model for convenience::

    def train(train_loader, val_loader, model, loss_criterion, optimizer, device, num_epochs=10):
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # print(batch.num_nodes)
                model.train()
                batch = batch.to(device)
                optimizer.zero_grad()

                pred_sim = model(batch.x_s, batch.edge_index_s, batch.x_t, batch.edge_index_t)
                loss = loss_criterion(pred_sim, batch.graph_sim)
                # Compute Gradients via Backpropagation
                loss.backward()
                # Update Parameters
                optimizer.step()
                train_losses.append(loss.item())

            for batch_idx, val_batch in enumerate(val_loader):
                model.eval()
                with torch.no_grad():
                    val_batch = val_batch.to(device)
                    pred_sim = model(val_batch.x_s, val_batch.edge_index_s, 
                            val_batch.x_t, val_batch.edge_index_t)
                    val_loss = loss_criterion(pred_sim, val_batch.graph_sim)
                    val_losses.append(val_loss.item())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
        
            # Printing Epoch Summary
            print(f"Epoch: {epoch+1}/{num_epochs} | Train MSE: {loss} | Validation MSE: {val_loss}")

With everything in place above, we train our model::
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = graphMatcher(av).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    train(train_loader, val_loader, model, criterion, optimizer, device)