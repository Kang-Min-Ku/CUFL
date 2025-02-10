import os
import pdb
import time
import torch
import random
import numpy as np

import networkx as nx
import metispy as metis

import torch_geometric
from torch_geometric.data import Data
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from ogb.nodeproppred import PygNodePropPredDataset

from utils import LargestConnectedComponents

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

mode = '' 
data_path = '../../../dataset'
seed = 1234
clients = [10]
ratio_train = 0.2
#dataset = f'ogbn-arxiv_CC_total_{ratio_train}'
dataset = f"cora_disjoint_{ratio_train}"
to_dense = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients):
    os.makedirs(os.path.join(data_path, dataset, str(n_clients)), exist_ok=True)
    st = time.time()
    data = get_data(dataset)
    data = split_train(data, dataset)
    split_heterogeneous(n_clients, data, dataset)
    print(f'done ({time.time()-st:.2f})')

def get_data(dataset):
    st = time.time()

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = datasets.Planetoid(data_path, dataset, transform=T.NormalizeFeatures())
        data = data[0]
    elif dataset in [f'cora_disjoint_{ratio_train}', f'citeseer_disjoint_{ratio_train}', f'pubmed_disjoint_{ratio_train}']:
        dataset = dataset.split("_")[0]
        data = datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))
        data = data[0]
    elif dataset in [f'computers_disjoint_{ratio_train}', f'photo_disjoint_{ratio_train}']:
        dataset = dataset.split("_")[0]
        data = datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))
        data = data[0]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in [f'ogbn-arxiv_disjoint_{ratio_train}']:
        dataset = dataset.split("_")[0]
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))
        data = data[0]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    elif dataset in [f'ogbn-proteins_disjoint_{ratio_train}']:
        dataset = dataset.split("_")[0]
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToSparseTensor(attr='edge_attr', remove_edge_index=False)]))
        data = data[0]
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in [f"ogbn-products_disjoint_{ratio_train}"]:
        dataset = dataset.split("_")[0]
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToSparseTensor(attr='edge_attr', remove_edge_index=False)]))
        data = data[0]
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    print(f'{dataset} have been loaded ({time.time()-st:.2f} sec)')
    return data

def split_train(data, dataset):
    st=time.time()
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    torch_save(data_path,f'{dataset}{mode}/{n_clients}/train.pt', {
        'data': data
    })
    torch_save(data_path,f'{dataset}{mode}/{n_clients}/test.pt', {
        'data': data
    })
    torch_save(data_path,f'{dataset}{mode}/{n_clients}/val.pt', {
        'data': data
    })
    print(f'splition done, n_train:{n_train}, n_test:{n_test}, n_val:{len(val_indices)} ({time.time()-st:.2f} sec)')
    return data

def split_heterogeneous(n_clients, data, dataset):
    st = time.time()
    fast = False

    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges:{n_cuts} ({time.time()-st:.2f})')
        
    if to_dense:
        adj = to_dense_adj(data.edge_index)[0]

    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        if to_dense:
            client_adj = adj[client_indices][:, client_indices]
            client_edge_index, _ = dense_to_sparse(client_adj)
            client_edge_index = client_edge_index.T.tolist()
        else:
            for _index, _edge in enumerate(data.edge_index.T):
                if _edge[0].item() in client_indices and \
                    _edge[1].item() in client_indices:
                    client_edge_index.append([
                        client_indices.index(_edge[0].item()), 
                        client_indices.index(_edge[1].item())
                    ])
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]

        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index.t().contiguous(),
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )

        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path,f'{dataset}{mode}/{n_clients}/heterogeneous_partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id:{client_id}, iid, n_train_node:{client_num_nodes}, n_train_edge:{client_num_edges} ({time.time()-st:.2f})')
        st = time.time()

for n_clients in clients:
    generate_data(dataset=dataset, n_clients=n_clients)
