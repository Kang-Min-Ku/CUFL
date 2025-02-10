import re
import yaml
import os
import copy
import numpy as np
import pandas as pd
from itertools import product
from .parser import YamlParser

def spawn_config(path:str, header:str, base_config:YamlParser, params:dict):
    # If params have hierarchy, specify it with dot(.)
    # e.g. fedpub.filter
    # assume 3-level
    os.makedirs(path, exist_ok=True)
    keys = list(params.keys())
    values = list(params.values())
    if len(keys) != len(values):
        raise ValueError("Length of keys and values must be the same.")

    for param in product(*values):
        config = copy.deepcopy(base_config)
        for key, p in zip(keys, param):
            segments = key.split(".")
            if len(segments) == 1:
                config[key] = p
            elif len(segments) == 2:
                config[segments[0]][segments[1]] = p
            elif len(segments) == 3:
                config[segments[0]][segments[1]][segments[2]] = p
            elif len(segments) == 4:
                config[segments[0]][segments[1]][segments[2]][segments[3]] = p
            else:
                raise ValueError("Length of segments must be less than 4.")
        config_file = os.path.join(path, f"{header}_{len(os.listdir(path))}.yaml")
        config.dump(config_file)
    
def spawn_script(config_path, output_file, header, base_command="python3 main.py --config ", identifier="progress.log", ):
    config_file = os.listdir(config_path)
    with open(output_file, "w") as fd:
        for file in config_file:
            if file.startswith(header):
                command = base_command + os.path.join(config_path, file) 
                fd.write(command + "\n")
                fd.write(f"echo {command} >> {identifier}\n")

def distribute_spawn_script(num_distribute, config_path, output_file, header, base_command="python3 main.py --config ", identifier="progress.log"):
    fds = []
    config_files = os.listdir(config_path)
    for i in range(num_distribute):
        fds.append(open(f"{i}_{output_file}", "w"))
    for idx, file in enumerate(config_files):
        if file.startswith(header):
            command = base_command + os.path.join(config_path, file)
            fds[idx % len(fds)].write(command + "\n")
            fds[idx % len(fds)].write(f"echo {command} >> {identifier}\n")

# def spec_hyperparam(df:pd.DataFrame, spec_list, identifier, output_path, config_file="config.yaml"):
#     spec_dict = {}
#     for trial in df[identifier]:
#         with open(os.path.join(output_path, trial, config_file), "r") as fd:
#             config = yaml.load(fd, Loader=yaml.FullLoader)
#         for spec in spec_list:
#             if spec not in spec_dict:
#                 spec_dict[spec] = []
#             split = spec.split(".")
#             if len(split) == 1:
#                 spec_dict[spec].append(config[spec])
#             elif len(split) == 2:
#                 spec_dict[spec].append(config[split[0]][split[1]])
#             elif len(split) == 3:
#                 spec_dict[spec].append(config[split[0]][split[1]][split[2]])
#             spec_dict[spec].append(config[spec])
#     df = pd.concat([df, pd.DataFrame(spec_dict)], axis=1)
#     return df
