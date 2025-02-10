import torch
import torch.nn as nn

class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return torch.sigmoid(value) if sigmoid else value
    
class ConsineDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        z_norm = z / z.norm(dim=-1, keepdim=True)
        value = (z_norm[edge_index[0]] * z_norm[edge_index[1]]).sum(dim=-1)
        #return torch.sigmoid(value) if sigmoid else value
        return (value + 1) / 2 if sigmoid else value # NOTE: Why can (value+1)/2 replace sigmoid? Value interval is [-1,1] in linear