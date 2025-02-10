import copy
import os
import inspect
import torch
import torch.nn as nn
from torch_geometric.data import Data
from .util import set_state_dict

def get_optimizer_state(optimizer):
    state = {}
    for p_key, p_value in optimizer.state_dict()["state"].items():
        state[p_key] = {}
        for name, value in p_value.items():
            if not torch.is_tensor(value):
                continue
            state[p_key][name] = value.clone().detach().cpu().numpy()
    return state

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

def select_optimizer(optimizer, parameters, args, **kwargs):
    if optimizer == "adam":
        optim_params = inspect.signature(torch.optim.Adam.__init__).parameters.keys()
        if args.optim_param is not None:
            optim_args = {k:args["optim_param"][k] for k in optim_params if k in args["optim_param"].keys()}
        else:
            optim_args = {}
        optim_args["params"] = parameters
        if "lr" not in optim_args.keys():
            optim_args["lr"] = args.base_lr
        optim_args = {**optim_args, **kwargs}
        optimizer = torch.optim.Adam(**optim_args)
    elif optimizer == "sgd":
        optim_params = inspect.signature(torch.optim.SGD.__init__).parameters.keys()
        if args.optim_param is not None:
            optim_args = {k:args["optim_param"][k] for k in optim_params if k in args["optim_param"].keys()}
        else:
            optim_args = {}
        optim_args["params"] = parameters
        if "lr" not in optim_args.keys():
            optim_args["lr"] = args.base_lr
        optim_args = {**optim_args, **kwargs}
        optimizer = torch.optim.SGD(**optim_args)
    return optimizer

def select_loss(loss_func, args):
    if loss_func == "cross_entropy":
        loss_params = inspect.signature(nn.CrossEntropyLoss.__init__).parameters.keys()
        if args.loss_param is not None:
            loss_args = {k:args["loss_param"][k] for k in loss_params if k in args["loss_param"].keys()}
        else:
            loss_args = {}
        loss_func = nn.CrossEntropyLoss(**loss_args)

    return loss_func

def load_pretrained(model, path, file, gpu_id):
    state_dict = torch.load(os.path.join(path, file))
    set_state_dict(model, state_dict, gpu_id)

    return model

def data_to_tgdataset(x, edge_index, edge_weight, gpu_id=None):
    data = Data(x, edge_index.long(), edge_weight)
    if gpu_id is not None:
        data = data.to(gpu_id)
    return data