import os
import sys
import yaml
import argparse
import secrets
import inspect
from datetime import datetime
from federated.utils.parser import YamlParser
from federated.utils.util import *
from federated.utils.logger import Logger
from federated.data.loader import DataLoader
from federated.setup.multiprocs import ParentProcess
from datetime import datetime

def set_config(args):
    token = None
    if args.datetoken:
        current_date = datetime.now()
        token = current_date.strftime("%Y.%m.%d.%H:%M:%S")
    # Create dir according to config
    for k,v in args.__dict__.items():
        if k in ["base_path", "data_path", "analysis_path"]:
            continue
        try:
            if isinstance(v, str) and k.endswith(args.path_suffix):
                if token is not None:
                    v = os.path.join(args.base_path, args.task, token, v)
                else:
                    v = os.path.join(args.base_path, args.task, v)
                args[k] = v
                os.makedirs(v, exist_ok=True)
        except:
            continue
    if args.tuning_analysis:
        os.makedirs(args.tuning_config["analyze_result_path"], exist_ok=True)
    if token is not None:
        args.config_path = os.path.join(args.base_path, args.task, token, "config.yaml")
    else:
        args.config_path = os.path.join(args.base_path, args.task, "config.yaml")
    with open(args.config_path, "w") as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)
    #logger
    log_file = "log.log" if args.log_file is None else args.log_file
    logger = Logger()
    logger.set_basic_config(filename=os.path.join(args.log_path, log_file))
    logger.set_logger()
    args.logger = logger
    
    if args.pretrain:
        msg = "Mode: Pretrain\n"
        print_and_log(msg, args.logger)
    else:
        msg = "Mode: Test\n"
        print_and_log(msg, args.logger)
    #dataset
    if args.task in ["cora_disjoint_0.2", "cora_overlapping_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients
        args.dist = "heterogeneous"
        args.num_feat = 1433
        args.num_class = 7
    elif args.task in ["citeseer_disjoint_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients 
        args.dist = "heterogeneous"
        args.num_feat = 3703
        args.num_class = 6
    elif args.task in ["pubmed_disjoint_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients
        args.dist = "heterogeneous"
        args.num_feat = 500
        args.num_class = 3
    elif args.task in ["ogbn-arxiv_disjoint_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients
        args.dist = "heterogeneous"
        args.num_feat = 128
        args.num_class = 40
    elif args.task in ["computers_disjoint_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients
        args.dist = "heterogeneous"
        args.num_feat = 767
        args.num_class = 10
    elif args.task in ["photo_disjoint_0.2"]:
        args.num_clients = 10 if args.num_clients is None else args.num_clients
        args.dist = "heterogeneous"
        args.num_feat = 745
        args.num_class = 8

    try:
        dataset = args.task.split("_")[0]
        args.curriculum["pretrained_header"] = f"fedprox_pretrained_{dataset}_client_{args.num_clients}_"
    except AttributeError:
        pass
    #loader
    args.loader = DataLoader
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    #model
    if args.model == "gcn":
        from federated.model.GCN import GCN
        args.model = GCN
    elif args.model == "maskedgcn":
        from federated.model.maskedGCN import MaskedGCN
        args.model = MaskedGCN
    elif args.model == "lwmaskedgcn":
        from federated.model.lwMaskedGCN import MaskedGCN
        args.model = MaskedGCN
    else:
        print('incorrect model was given: {}'.format(args.model))
        exit(1)

    if not hasattr(args, "curriculum"):
        pass
    elif args.curriculum["decoder"] == "cosine":
        from federated.model.decoder import ConsineDecoder
        args.curriculum["decoder"] = ConsineDecoder
    elif args.curriculum["decoder"] == "inner":
        from federated.model.decoder import InnerProductDecoder
        args.curriculum["decoder"] = InnerProductDecoder
    else:
        print('incorrect curriculum decoder was given: {}'.format(args.curriculum["decoder"]))
        exit(1)

    if args.tuning_analysis:
        args.tuning_config["analyze_result_path"] = os.path.join(args.tuning_config["analyze_result_path"], args.task.split("_")[0])
        args.tuning_config["analyze_result_file"] = f"{args.framework}_{args.task.split('_')[0]}_analysis.log"

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    args = set_config(YamlParser(parser.parse_args().config))

    if args.framework == "cufl":
        from federated.framework.cufl.client import Client
        from federated.framework.cufl.server import Server
    elif args.framework == "fedprox":
        from federated.framework.fedprox.client import Client
        from federated.framework.fedprox.server import Server
    else:
        print('incorrect framework was given: {}'.format(args.framework))
        exit(1)

    with open(os.path.join(os.path.dirname(args.config_path), "member.py"), "w+") as fd:
        fd.write(inspect.getsource(Client))
        fd.write(inspect.getsource(Server))

    pp = ParentProcess(args, Server, Client)
    pp.start()