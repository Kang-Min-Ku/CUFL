import time
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from ...utils.util import get_state_dict, set_state_dict, convert_np_to_tensor, print_and_log, fix_seed
from ...utils.torchutil import select_optimizer, torch_save, torch_load, load_pretrained, data_to_tgdataset

from ..module import BaseClient
from ...model.curriculum import SPCL
from ...model.scheduler import VectorGSS

class Client(BaseClient):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = self.args.model(self.args.num_feat, self.args.num_dims, self.args.num_class, self.args.l1, self.args).cuda(self.gpu_id)
        self.parameters = list(self.model.parameters())

    def init_state(self):
        self.proxy_data = self.sd["proxy"].cuda(self.gpu_id)
        self.optimizer = select_optimizer(self.args.optimizer, self.parameters, self.args)
        # NOTE: lambda scale scheduler
        self.scale_scheduler = VectorGSS(self.args.proxysim["scheduler"]["init_scale"],
                                                    self.args.proxysim["scheduler"]["window_size"],
                                                    self.args.proxysim["scheduler"]["patience"],
                                                    self.args.proxysim["scheduler"]["varying_factor"],
                                                    self.args.proxysim["scheduler"]["max_scale"],
                                                    self.args.proxysim["scheduler"]["min_scale"],
                                                    self.args.proxysim["scheduler"]["prefer_larger"])
        # NOTE: setup CL component
        pretrained_file = f'{self.args.curriculum["pretrained_header"]}{self.client_id}.{self.args.curriculum["pretrained_extension"]}'
        self.pretrained_model = self.args.model(self.args.num_feat, self.args.num_dims, self.args.num_class, self.args.l1, self.args).cuda(self.gpu_id)
        self.pretrained_model = load_pretrained(self.pretrained_model,
                                          self.args.curriculum["pretrained_path"], pretrained_file,
                                          self.gpu_id)
        data = next(iter(self.loader.train_loader)).to(self.gpu_id)
        out = self.pretrained_model(data)
        difficulty = F.cross_entropy(out, data.y, reduction="none").detach()
        norm_difficulty = torch.exp(self.args.curriculum["confidence_norm_scale"]*difficulty)
        if self.args.curriculum["confidence_norm_method"] == "min":
            norm_difficulty = norm_difficulty / norm_difficulty.min()
        elif self.args.curriculum["confidence_norm_method"] == "sum":
            norm_difficulty = norm_difficulty / norm_difficulty.sum()
        n1 = norm_difficulty[data.edge_index[0]]
        n1[~data.train_mask[data.edge_index[0]]] = 1.
        n2 = norm_difficulty[data.edge_index[1]]
        n2[~data.train_mask[data.edge_index[1]]] = 1.
        self.edge_confidence = n1 * n2
        
        self.cl_model = SPCL(
            max_edges= self.sd["max_edges"],
            num_edges=data.num_edges if self.args.curriculum["loss_type"] == "increase" else 2*data.num_edges,
            gpu_id=self.gpu_id,
            structure_decoder=self.args.curriculum["decoder"](),
            args = self.args
        ).cuda(self.gpu_id)
        self.proxy_cl_model = SPCL(
            max_edges = self.sd["proxy_max_edges"],
            num_edges=self.proxy_data.num_edges if self.args.curriculum["loss_type"] == "increase" else 2*self.proxy_data.num_edges,
            gpu_id=self.gpu_id,
            structure_decoder=self.args.curriculum["decoder"](),
            args = self.args
        ).cuda(self.gpu_id)
        self.spcl_optimizer = select_optimizer(self.args.curriculum["optimizer"], self.cl_model.parameters(), self.args, **self.args.curriculum["optim_param"])
        self.proxy_spcl_optimizer = select_optimizer(self.args.curriculum["optimizer"], self.proxy_cl_model.parameters(), self.args, **self.args.curriculum["proxy_optim_param"]) 
        # NOTE: Warm up
        with torch.no_grad():
            full_adj = torch.ones(data.edge_index.shape[-1]).cuda(self.gpu_id)
            
            z = self.pretrained_model(data, get_feature=True)
        self.pd = self.args.curriculum["warmup_pd"]

        for _ in range(self.args.curriculum["warmup_epochs"]):
            loss = self.train_spcl(self.cl_model, self.spcl_optimizer,
                            z, data.edge_index, full_adj,
                            self.pd, self.args.curriculum["loss_type"], self.args.curriculum["beta"],
                            is_proxy=False)

        msg = f"Client {self.client_id} Warmup complete"
        print_and_log(msg, self.logger, self.args.verbose_print_client, self.args.verbose_log_client)
        self.log = {
            'lr': [],
            'epoch_local_train_loss': [],'epoch_local_train_acc': [],
            'round_local_train_loss': [],'round_local_train_acc': [],
            'epoch_local_valid_loss': [],'epoch_local_valid_acc': [],
            'round_local_valid_loss': [],'round_local_valid_acc': [],
            'epoch_local_test_loss': [],'epoch_local_test_acc': [],
            'round_local_test_loss': [],'round_local_test_acc': [],
            'epoch_global_valid_loss': [],'epoch_global_valid_acc': [],
            'round_global_valid_loss': [],'round_global_valid_acc': [],
            'epoch_global_test_loss': [],'epoch_global_test_acc': [],
            'round_global_test_loss': [],'round_global_test_acc': [],
            "round_sparsity": [], "epoch_sparsity": []
        }
        if not self.args.use_mask:
            with torch.no_grad():
                for name, param in self.model.state_dict().items():
                    if "mask" in name:
                        param.copy_(torch.ones_like(param))

    def save_state(self):
        torch_save(self.args.checkpoint_path, f'{self.client_id}_state.pt', {
            "optimizer": self.optimizer.state_dict(),
            "model": get_state_dict(self.model),
            "pretrained_model": get_state_dict(self.pretrained_model),
            'log': self.log,
            "cl_model": self.cl_model.get_state(),
            "proxy_cl_model": self.proxy_cl_model.get_state(),
            "spcl_optimizer": self.spcl_optimizer.state_dict(),
            "proxy_spcl_optimizer": self.proxy_spcl_optimizer.state_dict(),
            "edge_confidence": self.edge_confidence,
            "pd": self.pd,
            "scale_scheduler": self.scale_scheduler
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpoint_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        set_state_dict(self.pretrained_model, loaded['pretrained_model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        if not self.args.use_mask:
            with torch.no_grad():
                for name, param in self.model.state_dict().items():
                    if "mask" in name:
                        param.copy_(torch.ones_like(param))
        self.log = loaded['log']
        self.cl_model.set_state(loaded["cl_model"], self.gpu_id)
        self.proxy_cl_model.set_state(loaded["proxy_cl_model"], self.gpu_id)
        self.spcl_optimizer.load_state_dict(loaded["spcl_optimizer"])
        self.proxy_spcl_optimizer.load_state_dict(loaded["proxy_spcl_optimizer"])
        self.edge_confidence = loaded["edge_confidence"].cuda(self.gpu_id)
        self.pd = loaded["pd"]
        self.scale_scheduler = loaded["scale_scheduler"]

    def update(self, update):
        self.prev_weight = convert_np_to_tensor(update['model'], self.gpu_id)
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def train(self):
        st = time.time()
        valid_global_acc, valid_global_loss = self.validate(mode="global")
        valid_local_acc, valid_local_loss = self.validate(mode="local")
        test_global_acc, test_global_loss = self.evaluate(mode="global")
        test_local_acc, test_local_loss = self.evaluate(mode="local")

        self.scale_scheduler.evaluate(valid_local_loss)
        
        msg = f"client: {self.client_id} round: {self.curr_round+1}, epochs: {0} "
        msg += f"valid global loss: {valid_global_loss:.4f}, valid global acc: {valid_global_acc:.4f}, "
        msg += f"valid local loss: {valid_local_loss:.4f}, valid local acc: {valid_local_acc:.4f}, "
        msg += f"test global loss: {test_global_loss:.4f}, test global acc: {test_global_acc:.4f}, "
        msg += f"test local loss: {test_local_loss:.4f}, test local acc: {test_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s) lambda: {self.scale_scheduler.scale}"
        print_and_log(msg, self.logger, self.args.verbose_print_client, self.args.verbose_log_client)

        self.log["epoch_global_valid_acc"].append(valid_global_acc)
        self.log["epoch_global_valid_loss"].append(valid_global_loss)
        self.log["epoch_local_valid_acc"].append(valid_local_acc)
        self.log["epoch_local_valid_loss"].append(valid_local_loss)
        self.log["epoch_global_test_acc"].append(test_global_acc)
        self.log["epoch_global_test_loss"].append(test_global_loss)
        self.log["epoch_local_test_acc"].append(test_local_acc)
        self.log["epoch_local_test_loss"].append(test_local_loss)

        self.masks = []
        for name, param in self.model.state_dict().items():
            if "mask" in name and self.args.mask_rank == -1:
                self.masks.append(param)

        if self.args.mask_rank != -1:
            for module in self.model.children():
                self.masks.append(module.mask)

        # NOTE: CL select edge to train
        data = next(iter(self.loader.train_loader)).to(self.gpu_id)
        masked_edge_index, masked_edge_weight = self.predict_spcl(self.cl_model, data, False)
        with torch.no_grad():
            edge_mask = self.cl_model.s_mask > self.args.curriculum["predict_mask_threshold"]
        if self.args.curriculum["use_edge_confidence"]:
            masked_edge_weight /= self.edge_confidence[edge_mask]
    
        for epoch in range(self.args.num_epochs):
            st = time.time()
            self.model.train()
            num_train = 0
            train_loss = []
            train_acc = 0.

            for i, batch in enumerate(self.loader.train_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda(self.gpu_id)
                masked_batch = data_to_tgdataset(batch.x, masked_edge_index, masked_edge_weight, self.gpu_id)
                y_hat = self.model(masked_batch)
                loss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
                
                for name, param in self.model.state_dict().items():
                    if self.args.use_mask and 'mask' in name and self.args.mask_rank == -1:
                        loss += torch.norm(param.float(), 1) * self.args.l1
                    elif 'conv' in name or 'classifier' in name:
                        if self.curr_round > 0:
                            loss += torch.norm(param.float()-self.prev_weight[name], 2) * self.args.loc_l2
                if self.args.mask_rank != -1:
                    for module in self.model.children():
                        loss += torch.norm(module.mask.float(), 1) * self.args.l1

                loss.backward()
                self.optimizer.step()

                num_train += batch.train_mask.sum().item()
                train_loss.append(loss.item())
                train_acc += (y_hat[batch.train_mask].argmax(dim=1) == batch.y[batch.train_mask]).sum().item()
            ######################## masking #########################################
            num_active = 0
            num_total = 1
            for mask in self.masks:
                pruned = torch.abs(mask) < self.args.l1 # l1 ??
                mask = torch.ones(mask.shape).cuda(self.gpu_id).masked_fill(pruned, 0)
                num_active += torch.sum(mask)
                _num_total = 1
                for s in mask.shape:
                    _num_total *= s
                num_total += _num_total
            if isinstance(num_active, torch.Tensor):
                sparsity = ((num_total - num_active) / num_total).item()
            else:
                sparsity = 0.
            ######################## masking #########################################

            valid_global_acc, valid_global_loss = self.validate(mode="global")
            valid_local_acc, valid_local_loss = self.validate(mode="local")
            test_global_acc, test_global_loss = self.evaluate(mode="global")
            test_local_acc, test_local_loss = self.evaluate(mode="local")

            train_loss = np.mean(train_loss)
            train_acc /= num_train

            msg = f"client: {self.client_id} round: {self.curr_round+1}, epochs: {epoch+1} "
            msg += f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, "
            msg += f"valid global loss: {valid_global_loss:.4f}, valid global acc: {valid_global_acc:.4f}, "
            msg += f"valid local loss: {valid_local_loss:.4f}, valid local acc: {valid_local_acc:.4f} "
            msg += f"test global loss: {test_global_loss:.4f}, test global acc: {test_global_acc:.4f}, "
            msg += f"test local loss: {test_local_loss:.4f}, test local acc: {test_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s) lambda: {self.scale_scheduler.scale}"
            print_and_log(msg, self.logger, self.args.verbose_print_client, self.args.verbose_log_client)

            if self.args.tuning_analysis and \
                (self.curr_round+1==self.args.num_rounds and epoch+1==self.args.num_epochs):
                with open(os.path.join(self.args.tuning_config["analyze_result_path"], self.args.tuning_config["analyze_result_file"]), "a+") as fd:
                    fd.write(f"{self.args.checkpoint_path}\n")
                    fd.write(msg)
                    fd.write("\n")

            self.log["epoch_local_train_loss"].append(train_loss)
            self.log["epoch_local_train_acc"].append(train_acc)
            self.log["epoch_global_valid_acc"].append(valid_global_acc)
            self.log["epoch_global_valid_loss"].append(valid_global_loss)
            self.log["epoch_local_valid_acc"].append(valid_local_acc)
            self.log["epoch_local_valid_loss"].append(valid_local_loss)
            self.log["epoch_global_test_acc"].append(test_global_acc)
            self.log["epoch_global_test_loss"].append(test_global_loss)
            self.log["epoch_local_test_acc"].append(test_local_acc)
            self.log["epoch_local_test_loss"].append(test_local_loss)
            self.log["epoch_sparsity"].append(sparsity)

        self.log["round_local_train_loss"].append(self.log["epoch_local_train_loss"][-1])
        self.log["round_local_train_acc"].append(self.log["epoch_local_train_acc"][-1])
        self.log["round_global_valid_acc"].append(self.log["epoch_global_valid_acc"][-1])
        self.log["round_global_valid_loss"].append(self.log["epoch_global_valid_loss"][-1])
        self.log["round_local_valid_acc"].append(self.log["epoch_local_valid_acc"][-1])
        self.log["round_local_valid_loss"].append(self.log["epoch_local_valid_loss"][-1])
        self.log["round_global_test_acc"].append(self.log["epoch_global_test_acc"][-1])
        self.log["round_global_test_loss"].append(self.log["epoch_global_test_loss"][-1])
        self.log["round_local_test_acc"].append(self.log["epoch_local_test_acc"][-1])
        self.log["round_local_test_loss"].append(self.log["epoch_local_test_loss"][-1])
        self.log["round_sparsity"].append(sparsity)

        self.save_log()

    def transfer_to_server(self):
        proxy_recon_edge = self.proxy_cl_model.graph_recon_degree.cpu().detach().numpy() / self.proxy_cl_model.num
        if self.args.curriculum["transfer_mask_method"] == "ratio":
            proxy_recon_edge[proxy_recon_edge < np.quantile(proxy_recon_edge, self.args.curriculum["transfer_mask_ratio"])] = 0.
        elif self.args.curriculum["transfer_mask_method"] == "threshold":
            proxy_recon_edge[proxy_recon_edge < self.args.curriculum["transfer_mask_threshold"]] = 0.

        self.sd[self.client_id] = {
            "weights": get_state_dict(self.model),
            "train_size": len(self.loader.partition),
            "proxy_recon_edge": proxy_recon_edge,
            "scale": self.scale_scheduler.scale
        }

    def train_spcl(self, spcl_model, optimizer, z, edge_index, full_adj=None, pd=1., loss_type="increase", beta=1., is_proxy=True):
        spcl_model.train()
        optimizer.zero_grad()
        if full_adj is None:
            full_adj = torch.ones(edge_index.shape[-1]).cuda(self.gpu_id)
        loss = spcl_model.recon_loss(z, edge_index, pd, full_adj, loss_type, beta)
        msg = ""
        if hasattr(self, 'curr_round'):
            msg = f"client: {self.client_id} round: {self.curr_round+1} "
        else:
            msg = f"client: {self.client_id} round: init "
        if is_proxy:
            msg += f"spcl loss: {loss.item()} proxy smask sum: {spcl_model.s_mask.sum().item()}"
        else:
            msg += f"spcl loss: {loss.item()} smask sum: {spcl_model.s_mask.sum().item()}"
        print_and_log(msg, self.logger, not self.args.verbose_print_client, self.args.verbose_log_client)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p in spcl_model.parameters():
                p.clamp_(min=0, max=1)
        return loss

    def train_cl_model(self, cl_model, model, optimizer, curr_round, is_proxy):
        if not is_proxy and curr_round % self.args.curriculum["train_cycle"] > 0:
            return
        elif is_proxy and curr_round % self.args.curriculum["proxy_train_cycle"] > 0:
            return
        data = None
        masked_data = None
        with torch.no_grad():
            if is_proxy:
                data = self.proxy_data
                masked_proxy_edge_index, masked_proxy_edge_weight = self.predict_spcl(cl_model, data, False)
                masked_data = data_to_tgdataset(data.x, masked_proxy_edge_index, masked_proxy_edge_weight, self.gpu_id)
                z = model(masked_data, get_feature=True)
            else:
                data = next(iter(self.loader.train_loader)).to(self.gpu_id)
                masked_edge_index, masked_edge_weight = self.predict_spcl(cl_model, data, False)
                edge_mask = cl_model.s_mask > self.args.curriculum["predict_mask_threshold"]
                if self.args.curriculum["use_edge_confidence"]:
                    masked_edge_weight /= self.edge_confidence[edge_mask]
                masked_data = data_to_tgdataset(data.x, masked_edge_index, masked_edge_weight, self.gpu_id)
                z = model(masked_data, get_feature=True)
        
        full_adj = torch.ones(data.edge_index.shape[-1]).cuda(self.gpu_id)
        for _ in range(self.args.curriculum["train_epochs"]):
            self.train_spcl(cl_model, optimizer,
                            z, data.edge_index, full_adj,
                            self.pd, self.args.curriculum["loss_type"], self.args.curriculum["beta"],
                            is_proxy=is_proxy)
            
        _ = self.predict_spcl(cl_model, data, True)
            
    def predict_spcl(self, spcl_model, data, update_accumul=True):
        spcl_model.eval()
        with torch.no_grad():
            masked_edge_index, masked_edge_weight = spcl_model.predict(data.edge_index, threshold=self.args.curriculum["predict_mask_threshold"], update_accumul=update_accumul)
            return masked_edge_index, masked_edge_weight

    def update_pd(self, curr_round):
        if self.args.curriculum["pd_update_rule"] == "1":
            self.pd = self.args.curriculum["base_pd"] / (self.args.num_rounds*2//3+1-curr_round) \
                        if curr_round < self.args.num_rounds*2//3 else self.args.curriculum["base_pd"]
        elif self.args.curriculum["pd_update_rule"] == "descend_1":
            self.pd = self.args.curriculum["base_pd"] - self.args.curriculum["base_pd"] / (self.args.num_rounds*2//3+1-curr_round) \
                        if curr_round < self.args.num_rounds*2//3 else self.args.curriculum["base_pd"] #descending
        elif self.args.curriculum["pd_update_rule"] == "2":
            self.pd = self.args.curriculum["base_pd"] / (self.args.num_rounds+1-curr_round) \
                        if curr_round < self.args.num_rounds else self.args.curriculum["base_pd"]
        elif self.args.curriculum["pd_update_rule"] == "3":
            self.pd = self.args.curriculum["base_pd"] * curr_round/self.args.num_rounds + self.args.curriculum["warmup_pd"]
        elif self.args.curriculum["pd_update_rule"] == "valid":
            NotImplementedError

    def on_receive_message(self, curr_round):
        self.curr_round = curr_round
        self.update(self.sd[f'adaptive_{self.client_id}' \
            if (f'adaptive_{self.client_id}' in self.sd) else 'global'])
        self.global_weight = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)

    def cl_update(self, curr_round):
        self.update_pd(curr_round)
        self.train_cl_model(self.cl_model, self.model, self.spcl_optimizer, curr_round, is_proxy=False)
        self.train_cl_model(self.proxy_cl_model, self.model, self.proxy_spcl_optimizer, curr_round, is_proxy=True)

    def on_round_begin(self, curr_round):
        self.train()

    def on_round_complete(self, curr_round):
        self.transfer_to_server()

    def local_job(self, curr_round):
        self.on_receive_message(curr_round)
        self.on_round_begin(curr_round)
        self.cl_update(curr_round)
        self.on_round_complete(curr_round)