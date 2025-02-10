import os
import copy
import pdb
import glob
import json
import logging
import time
import torch
import random
import numpy as np
from collections import OrderedDict
from .pdb import ForkedPdb

def str2bool(v):
  return v.lower() in ['true', 't']

def shuffle(seed, x, y):
    idx = np.arange(len(x))
    random.seed(seed)
    random.shuffle(idx)
    return [x[i] for i in idx], [y[i] for i in idx]

def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)

def exists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))

def join_glob(base_dir, filename):
    return glob.glob(os.path.join(base_dir, filename))

def remove_if_exist(base_dir, filename):
    targets = join_glob(base_dir, filename)
    if len(targets)>0:
        for t in targets:
            os.remove(t)

def debugger():
    ForkedPdb().set_trace()

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict

def get_weights(model):
    return {"model":get_state_dict(model)}

def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask, model=model.state_dict())
    model.load_state_dict(state_dict)

def set_weights(model, state_dict, gpu_id):
    set_state_dict(model, state_dict, gpu_id)
    
def convert_tensor_to_np(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])

def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict

def convert_np_to_tensor_cpu(state_dict):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        _state_dict[k] = torch.tensor(v)
    return _state_dict

def print_and_log(msg, logger, verbose_print=True, verbose_log=True, level=logging.INFO):
    if verbose_print:
        print(msg)
    if verbose_log:
        if level == logging.INFO:
            logger.info(msg)
        elif level == logging.DEBUG:
            logger.debug(msg)
        elif level == logging.WARNING:
            logger.warning(msg)
        elif level == logging.ERROR:
            logger.error(msg)
        elif level == logging.CRITICAL:
            logger.critical(msg)
        else:
            raise ValueError(f"level {level} is not supported")

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)