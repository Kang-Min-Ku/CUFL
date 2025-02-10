import time
import random
import numpy as np
from scipy.spatial.distance import cosine

import torch

from ...utils.util import set_state_dict, get_state_dict, print_and_log, fix_seed
from ...utils.torchutil import torch_save, torch_load
from ..module import BaseServer

class Server(BaseServer):

    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = self.args.model(self.args.num_feat, self.args.num_dims, self.args.num_class, self.args.l1, self.args).cuda(self.gpu_id)
        self.log = {
            'round_valid_acc': [], 'round_valid_lss': [],
            'round_test_acc': [], 'round_test_lss': [],
            'best_valid_round': 0, 'best_valid_acc': 0, 'test_acc': 0
        }
        self.sd['proxy'] = self.get_proxy_data(args.num_feat)
        self.sd["proxy_max_edges"] = self.sd["proxy"].num_edges
        local_edges = []
        temp_loader = self.args.loader(self.args)
        for i in range(self.args.num_clients):
            temp_loader.get_loader(i)
            local_edges.append(next(iter(temp_loader.train_loader)).num_edges)
        del temp_loader
        self.sd["max_edges"] = max(local_edges)

        self.update_lists = []
        self.sim_matrices = []

        self.num_connected = round(self.args.num_clients*self.args.fraction)
        self.avg_sim_matrix = np.zeros(shape=(self.num_connected, self.num_connected))

    def get_proxy_data(self, num_feat):
        import networkx as nx
        from torch_geometric.utils import from_networkx

        num_graphs = self.args.proxysim["num_proxy"]
        num_nodes = self.args.proxysim["num_nodes"]
        G = nx.random_partition_graph(
            [num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed)
        data = from_networkx(G)
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, num_feat))

        return data

    def on_round_begin(self, selected, curr_round):
        self.round_st = time.time()
        self.curr_round = curr_round
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        valid_acc, valid_loss = self.validate()
        test_acc, test_loss = self.evaluate()

        if self.log["best_valid_acc"] < valid_acc:
            self.log["best_valid_acc"] = valid_acc
            self.log["best_valid_round"] = self.curr_round
            self.log["test_acc"] = test_acc
            self.save_state()

        self.log["round_valid_acc"].append(valid_acc)
        self.log["round_valid_lss"].append(valid_loss)
        self.log["round_test_acc"].append(test_acc)
        self.log["round_test_lss"].append(test_loss)
        
        msg = f"round:{self.curr_round+1}, curr_valid_loss:{valid_loss:.4f}, curr_valid_acc:{valid_acc:.4f}, "
        msg += f"best_valid_acc: {self.log['best_valid_acc']:.4f}, test_acc:{test_acc:.4f},({time.time()-self.round_st:.2f}s)"
        print_and_log(msg, self.logger, self.args.verbose_print_server, self.args.verbose_log_server)
        
        self.save_log()

    def get_sim_matrix(self, local_proxy_recon_edges, scale):
        assert self.num_connected == len(local_proxy_recon_edges), "Number of participants is not equal to the number of proxy outputs"
        if isinstance(scale, (int, float)):
            scale = np.ones(self.num_connected) * scale
        elif isinstance(scale, list):
            scale = np.array(scale)
        scale = scale[:, np.newaxis]

        sim_matrix = np.empty(shape=(self.num_connected, self.num_connected))
        for i in range(self.num_connected):
            for j in range(self.num_connected):
                sim_matrix[i][j] = 1 - cosine(local_proxy_recon_edges[i], local_proxy_recon_edges[j])

        if self.args.proxysim["agg_metric"] == "exp":
            sim_matrix = np.exp(scale*sim_matrix)
        
        if self.args.proxysim["filter"]:
            for i in range(self.num_connected):
                mask = (sim_matrix[i] < sim_matrix[i].mean())
                sim_matrix[i][mask] = 0
        
        row_sum = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sum[:, np.newaxis]

        return sim_matrix
            
    def update(self, updated):
        st = time.time()
        local_weights = []
        local_proxy_recon_edges = []
        local_train_sizes = []
        local_scales = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]["weights"].copy())
            local_proxy_recon_edges.append(self.sd[c_id]["proxy_recon_edge"])
            local_train_sizes.append(self.sd[c_id]["train_size"])
            local_scales.append(self.sd[c_id]["scale"])
            del self.sd[c_id]

        msg = f"all clients have been uploaded their weights ({time.time()-st:.2f} s)"
        print_and_log(msg, self.logger, self.args.verbose_print_server, self.args.verbose_log_server)
        sim_matrix = self.get_sim_matrix(local_proxy_recon_edges, local_scales)

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        # NOTE: update global model
        self.set_weights(self.model, self.aggregate(local_weights, ratio, -1))

        msg = f"global model has been updated ({time.time()-st:.2f}s)"
        print_and_log(msg, self.logger, self.args.verbose_print_server, self.args.verbose_log_server)

        st = time.time()
        # NOTE: update sim_matrix
        for i, c_id in enumerate(updated):
            ratio = sim_matrix[i, :]
            # NOTE: aggregate & send
            aggregated = self.aggregate(local_weights, ratio, c_id)
            if f"adaptive_{c_id}" in self.sd:
                del self.sd[f"adaptive_{c_id}"]
            self.sd[f"adaptive_{c_id}"] = {
                "model": aggregated
            }
        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)

        msg = f"local models have been updated ({time.time()-st:.2f}s)"
        print_and_log(msg, self.logger, self.args.verbose_print_server, self.args.verbose_log_server)

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpoint_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'log': self.log,
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })
