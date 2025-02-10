import time
import os
import copy
import torch
import numpy as np
import torch.nn.functional as F

from ...utils.util import get_state_dict, set_state_dict, convert_np_to_tensor, print_and_log, fix_seed
from ...utils.torchutil import select_optimizer, torch_save, torch_load

from ..module import BaseClient

class Client(BaseClient):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = self.args.model(self.args.num_feat, self.args.num_dims, self.args.num_class, self.args.l1, self.args).cuda(self.gpu_id)
        self.pretrained_model = self.args.model(self.args.num_feat, self.args.num_dims, self.args.num_class, self.args.l1, self.args).cuda(self.gpu_id)
        self.best_pretrained_acc = 0.
        # debug
        if self.args.debug:
            fix_seed(self.args.seed)
            weight_file = self.args.debug_config["client_model_header"]
            if self.args.debug_config["order_by"] == "worker_id":
                weight_file += f"{self.worker_id}"
            weight_file += f".{self.args.debug_config['model_file_extension']}"
            set_state_dict(self.model,torch_load(self.args.debug_config["client_model_path"],weight_file),self.gpu_id)
        self.parameters = list(self.model.parameters()) 

    def init_state(self):
        self.optimizer = select_optimizer(self.args.optimizer, self.parameters, self.args)
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
        save_obj = {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        }
        if self.args.pretrain:
            save_obj["pretrained_model"] = get_state_dict(self.pretrained_model)
            save_obj["best_pretrained_acc"] = self.best_pretrained_acc
            save_obj["best_pretrained_param"] = self.best_pretrained_param
        torch_save(self.args.checkpoint_path, f'{self.client_id}_state.pt', save_obj)

    def load_state(self):
        loaded = torch_load(self.args.checkpoint_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        if not self.args.use_mask:
            with torch.no_grad():
                for name, param in self.model.state_dict().items():
                    if "mask" in name:
                        param.copy_(torch.ones_like(param))
        self.log = loaded['log']
        if self.args.pretrain:
            set_state_dict(self.pretrained_model, loaded["pretrained_model"], self.gpu_id)
            self.best_pretrained_acc = loaded["best_pretrained_acc"]
            self.best_pretrained_param = loaded["best_pretrained_param"]

    def update(self, update):
        self.prev_weight = convert_np_to_tensor(update['model'], self.gpu_id)
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def train(self):
        st = time.time()
        valid_global_acc, valid_global_loss = self.validate(mode="global")
        valid_local_acc, valid_local_loss = self.validate(mode="local")
        test_global_acc, test_global_loss = self.evaluate(mode="global")
        test_local_acc, test_local_loss = self.evaluate(mode="local")
        
        msg = f"client: {self.client_id} round: {self.curr_round+1}, epochs: {0} "
        msg += f"valid global loss: {valid_global_loss:.4f}, valid global acc: {valid_global_acc:.4f}, "
        msg += f"valid local loss: {valid_local_loss:.4f}, valid local acc: {valid_local_acc:.4f}, "
        msg += f"test global loss: {test_global_loss:.4f}, test global acc: {test_global_acc:.4f}, "
        msg += f"test local loss: {test_local_loss:.4f}, test local acc: {test_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)"
        print_and_log(msg, self.logger, self.args.verbose_print_client, self.args.verbose_log_client)

        self.log["epoch_global_valid_acc"].append(valid_global_acc)
        self.log["epoch_global_valid_loss"].append(valid_global_loss)
        self.log["epoch_local_valid_acc"].append(valid_local_acc)
        self.log["epoch_local_valid_loss"].append(valid_local_loss)
        self.log["epoch_global_test_acc"].append(test_global_acc)
        self.log["epoch_global_test_loss"].append(test_global_loss)
        self.log["epoch_local_test_acc"].append(test_local_acc)
        self.log["epoch_local_test_loss"].append(test_local_loss)

        #################################################################
        self.masks = []
        for name, param in self.model.state_dict().items():
            if "mask" in name and self.args.mask_rank == -1:
                self.masks.append(param)

        if self.args.mask_rank != -1:
            for module in self.model.children():
                self.masks.append(module.mask)

        #################################################################
        for epoch in range(self.args.num_epochs):
            st = time.time()
            self.model.train()
            num_train = 0
            train_loss = []
            train_acc = 0.

            for i, batch in enumerate(self.loader.train_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda(self.gpu_id)
                y_hat = self.model(batch)
                loss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
                ######################## regularization #########################################
                for name, param in self.model.state_dict().items():
                    if 'conv' in name or 'classifier' in name:
                        if self.curr_round > 0:
                            loss += torch.norm(param.float()-self.prev_weight[name], 2) * self.args.loc_l2
                ######################## regularization #########################################

                loss.backward()
                self.optimizer.step()

                num_train += batch.train_mask.sum().item()
                train_loss.append(loss.item())
                train_acc += (y_hat[batch.train_mask].argmax(dim=1) == batch.y[batch.train_mask]).sum().item()
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

            z = self.model(batch, get_feature=True)
            edge_index = batch.edge_index
            z_norm = z / z.norm(dim=-1, keepdim=True)
            value = (z_norm[edge_index[0]] * z_norm[edge_index[1]]).sum(dim=-1)
            value = (value + 1) / 2

            valid_global_acc, valid_global_loss = self.validate(mode="global")
            valid_local_acc, valid_local_loss = self.validate(mode="local")
            test_global_acc, test_global_loss = self.evaluate(mode="global")
            test_local_acc, test_local_loss = self.evaluate(mode="local")

            train_loss = np.mean(train_loss)
            train_acc /= num_train

            msg = f"client: {self.client_id} round: {self.curr_round+1}, epochs:{epoch+1} "
            msg += f"train loss:{train_loss:.4f}, train acc:{train_acc:.4f}, "
            msg += f"valid global loss:{valid_global_loss:.4f}, valid global acc:{valid_global_acc:.4f}, "
            msg += f"valid local loss:{valid_local_loss:.4f}, valid local acc:{valid_local_acc:.4f} "
            msg += f"test global loss:{test_global_loss:.4f}, test global acc:{test_global_acc:.4f}, "
            msg += f"test local loss:{test_local_loss:.4f}, test local acc:{test_local_acc:.4f}, lr:{self.get_lr()} ({time.time()-st:.2f}s)"
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

        if self.args.pretrain and valid_local_acc >= self.best_pretrained_acc:    
            self.best_pretrained_acc = test_local_acc
            self.best_pretrained_param = get_state_dict(self.model)

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
        self.sd[self.client_id] = {
            'weights': get_state_dict(self.model),
            'train_size': len(self.loader.partition),
        }

    def on_receive_message(self, curr_round):
        self.curr_round = curr_round
        self.update(self.sd[f'adaptive_{self.client_id}' \
            if (f'adaptive_{self.client_id}' in self.sd) else 'global'])
        self.global_weight = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)

    def on_round_begin(self, curr_round):
        self.train()
        self.transfer_to_server()

    def save_model(self, curr_round):
        if not os.path.exists(self.args.pretrain_config["pretrained_model_path"]):
            os.makedirs(self.args.pretrain_config["pretrained_model_path"], exist_ok=True)
        if self.args.pretrain and curr_round == self.args.num_rounds - 1:
            if self.args.pretrain_config["order_by"] == "client_id":
                dataset = self.args.task.split("_")[0]
                file = f"fedprox_pretrained_{dataset}_client_{self.args.num_clients}_{self.client_id}.pt"
            torch_save(self.args.pretrain_config["pretrained_model_path"], file, self.best_pretrained_param)

    def local_job(self, curr_round):
        self.on_receive_message(curr_round)
        self.on_round_begin(curr_round)
        self.save_model(curr_round)