import time
import os
import numpy as np
from collections import OrderedDict
from abc import abstractmethod, ABCMeta

import torch
import torch.nn.functional as F

from ..utils.util import save, fix_seed
from ..data.loader import DataLoader

class BaseServer(metaclass=ABCMeta):
    def __init__(self, args, sd, gpu_server):
        self.args = args
        self._args = vars(self.args)
        self.gpu_id = gpu_server
        self.sd = sd
        self.loader = self.args.loader(self.args, is_server=True)
        
        assert "logger" in args.__dict__.keys(), "Logger is not defined in args"
        self.logger = args.logger
        if "log_header" in self.args.__dict__.keys():
            self.log_header = self.args.log_header
        else:
            self.log_header = f"[{self.args.framework}]"
            self.log_header += f"[{self.args.model}]"
            self.log_header += f"[{self.args.task}]"
            self.log_header += f"[server] "

        if not self.args.debug:
            fix_seed(self.args.seed)

    def get_active(self, mask):
        active = np.absolute(mask) >= self.args.l1
        return active.astype(float)

    def aggregate(self, local_weights, ratio=None, client_id=None):
        st = time.time()
        aggr_theta = OrderedDict([(k,None) for k in local_weights[0].keys()])
        if ratio is not None:
            ratio = np.array(ratio)
            if len(ratio.shape) == 1:
                ratio = ratio[np.newaxis, :]
            num_layers = ratio.shape[0]
            for name, params in aggr_theta.items():
                # TODO: make general. Yet, model specific
                # get index
                if name.startswith("convs"):
                    idx = int(name.split('.')[1])
                    if idx >= num_layers:
                        idx = -1
                elif name.startswith("classifier"):
                    idx = -1
                    if not self.args.aggregate_classifier and client_id != -1:
                        aggr_theta[name] = local_weights[client_id][name]
                        continue
                if self.args.mask_aggr:
                    if 'mask' in name:
                        # get active
                        acti = [ratio[idx][i]*self.get_active(lw[name])+1e-8 for i, lw in enumerate(local_weights)]
                        # get element_wise ratio
                        elem_wise_ratio = acti/np.sum(acti, 0)
                        # perform element_wise aggr
                        aggr_theta[name] = np.sum([theta[name]*elem_wise_ratio[j] for j, theta in enumerate(local_weights)], 0)
                    else:
                        aggr_theta[name] = np.sum([theta[name]*ratio[idx][j] for j, theta in enumerate(local_weights)], 0)
                else:
                    aggr_theta[name] = np.sum([theta[name]*ratio[idx][j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1/len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        # self.logger.print(f'weight aggregation done ({round(time.time()-st, 3)} s)')
        return aggr_theta

    @torch.no_grad()
    def evaluate(self):
        if not self.args.eval_global:
            return 0, np.mean([0])

        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(self.loader.test_loader):
                batch = batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(batch, batch.test_mask)
                pred.append(y_hat[batch.test_mask])
                target.append(batch.y[batch.test_mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.num_class), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def validate(self):
        if not self.args.eval_global:
            return 0, np.mean([0])

        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(self.loader.valid_loader):
                batch = batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(batch, batch.val_mask)
                pred.append(y_hat[batch.val_mask])
                target.append(batch.y[batch.val_mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.num_class), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.model.eval()
        y_hat = self.model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1]
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def save_log(self):
        save(self.args.log_path, f'server.txt', {
            'log': self.log
        }) #'args': self._args,

    @abstractmethod
    def on_round_begin(self, selected, curr_round):
        raise NotImplementedError()
    
    @abstractmethod
    def on_round_complete(self, updated):
        raise NotImplementedError()

class BaseClient(metaclass=ABCMeta):
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = self.args.loader(self.args)
        #self.logger = Logger(self.args, self.gpu_id)
        assert "logger" in args.__dict__.keys(), "Logger is not defined in args"
        self.logger = args.logger
        self.log_header = ""

        if not self.args.debug:
            fix_seed(self.args.seed)
       
    def switch_state(self, client_id):
        self.client_id = client_id
        #self.loader.switch(client_id)
        self.loader.get_loader(client_id)
        if "log_header" in self.args.__dict__.keys(): #same with logger switch
            self.log_header = self.args.log_header
        else:
            self.log_header = f"[{self.args.framework}]"
            self.log_header += f"[{self.args.model}]"
            self.log_header += f"[{self.args.task}]"
            self.log_header += f"[client #{self.client_id}] "
        #self.logger.switch(client_id)
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpoint_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(self, mode='global'):
        if mode == 'global' and not self.args.eval_global:
            return 0, np.mean([0])

        if mode == 'global':    loader = self.loader.test_loader
        elif mode == 'local':   loader = self.loader.train_loader
        else:                   raise ValueError()
        
        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(loader):
                batch = batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(batch, batch.test_mask)
                pred.append(y_hat[batch.test_mask])
                target.append(batch.y[batch.test_mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.num_class), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def evaluate_neighbor(self):
        loader = self.loader.ne_loader
        
        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(loader):
                batch = batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(batch, batch.test_mask)
                pred.append(y_hat[batch.test_mask])
                target.append(batch.y[batch.test_mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.num_class), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def validate(self, mode='global', model=None):
        if mode == 'global' and not self.args.eval_global:
            return 0, np.mean([0])

        if mode == 'global':    loader = self.loader.valid_loader
        elif mode == 'local':   loader = self.loader.train_loader
        else:                   raise ValueError()

        with torch.no_grad():
            target, pred, loss = [], [], []
            for i, batch in enumerate(loader):
                batch = batch.cuda(self.gpu_id)
                y_hat, lss = self.validation_step(batch, batch.val_mask, model)
                pred.append(y_hat[batch.val_mask])
                target.append(batch.y[batch.val_mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.num_class), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def validation_step(self, batch, mask=None, model=None):
        y_hat = None
        if model is not None:
            model.eval()
            y_hat = model(batch)
        else:
            self.model.eval()
            y_hat = self.model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1]
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'log': self.log
        }) #'args': self._args,

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state
    
    @abstractmethod
    def local_job(self, curr_round):
        raise NotImplementedError()
