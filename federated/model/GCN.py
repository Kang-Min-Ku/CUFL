import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_feat=10, num_dims=128, num_class=10, l1=None, args=None):
        super().__init__()
        self.num_feat = num_feat
        self.num_dims = num_dims
        self.num_class = num_class
        self.args = args
        self.num_layers = self.args.num_layers

        self.convs = nn.ModuleDict()
        for i in range(self.num_layers):
            if i == 0:
                self.convs[str(i)] = GCNConv(self.num_feat, self.num_dims)
            else:
                self.convs[str(i)] = GCNConv(self.num_dims, self.num_dims)
                
        self.classifier = nn.Linear(self.num_dims, self.num_class)

    def forward(self, data, get_feature=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            x = self.convs[str(i)](x, edge_index, edge_weight)
            x = F.relu(x)
            if self.args.use_dropout and not self.args.debug:
                x = F.dropout(x, training=self.training)
        if get_feature: return x
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x