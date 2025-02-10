import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from .decoder import InnerProductDecoder
from ..utils.util import get_state_dict, set_state_dict

class SPCL(nn.Module): # NOTE: RCL suggested
    def __init__(self, max_edges, num_edges, gpu_id, structure_decoder=None, epsilon=1e-8, args=None):
        super().__init__()
        self.max_edges = max_edges
        self.num_edges = num_edges
        self._s_mask = nn.Parameter(torch.zeros(max_edges, dtype=torch.float))
        self.accumulated_s_mask = torch.zeros(num_edges, dtype=torch.float).cuda(gpu_id)
        self.graph_recon_degree = torch.zeros(num_edges, dtype=torch.float).cuda(gpu_id)
        self.structure_decoder = InnerProductDecoder() if structure_decoder is None else structure_decoder #REP: No backpropagation
        self.epsilon = epsilon
        self.num = self.epsilon
        self.args = args

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.s_mask)
        reset(self.structure_decoder)

    def get_state(self):
        return {
            "s_mask": get_state_dict(self),
            "accumulated_s_mask": self.accumulated_s_mask,
            "graph_recon_degree": self.graph_recon_degree,
            "num": self.num,
            "max_edges": self.max_edges,
            "num_edges": self.num_edges
        }
    
    def set_state(self, state, gpu_id):
        set_state_dict(self, state["s_mask"], gpu_id)
        self.accumulated_s_mask = state["accumulated_s_mask"].cuda(gpu_id)
        self.__setattr__("accumulated_s_mask", self.accumulated_s_mask)
        self.graph_recon_degree = state["graph_recon_degree"].cuda(gpu_id)
        self.__setattr__("graph_recon_degree", self.graph_recon_degree)
        self.num = state["num"]
        self.max_edges = state["max_edges"]
        self.num_edges = state["num_edges"]

    def recon_loss(self, z, edge_index, pd, full_adj, loss_type="increase", beta=1.):
        # NOTE: pd is personalization degree
        pred_struct = self.structure_decoder(z, edge_index).view(-1)
        if loss_type == "increase":
            structure_loss = torch.sum(
                self.s_mask * (pred_struct - full_adj) ** 2
            ) - pd * torch.sum(self.s_mask)
            # structure_loss += beta*torch.mean((self.s_mask - full_adj) ** 2)
        elif loss_type == "both":
            structure_loss = torch.mean(self.s_mask*pred_struct)
            structure_loss += beta*torch.mean(self.s_mask)
            structure_loss += pd*torch.mean((self.s_mask - full_adj) ** 2)

        
        return structure_loss
    
    @torch.no_grad()
    def predict(self, edge_index, threshold=0.5, update_accumul=True): 
        with torch.no_grad():
            mask = self.s_mask > threshold
            masked_edge_index = edge_index[:, mask]
            if update_accumul:
                self.graph_recon_degree = self.s_mask.detach().float() + self.graph_recon_degree
                self.accumulated_s_mask = mask.detach().float() + self.accumulated_s_mask
                self.num += 1
            masked_edge_weight = self.accumulated_s_mask[mask] / self.num
            
        return masked_edge_index, masked_edge_weight

    @property
    def s_mask(self):
        return self._s_mask[:self.num_edges]