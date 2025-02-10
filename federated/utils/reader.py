import re
import yaml
import os
import copy
import numpy as np
import pandas as pd
from itertools import product
from .parser import YamlParser

def parse_analysis_log(log_file,
                output_file=None,
                client_regex=r"client: (?P<client>\d).+test local acc: (?P<acc>0\.\d+)",
                distinguish_client_by="client",
                identifier_regex=r"/(?P<trial_name>[\d:\.]+)/",
                is_save=False
                ):
    result = {}
    trial_name = ""
    with open(log_file, "r") as fd:
        while True:
            line = fd.readline()
            if not line:
                break

            try:
                match = re.search(client_regex, line)
                if match is None:
                    trial_name = re.search(identifier_regex, line).groupdict()["trial_name"]
                    if trial_name not in result:
                        result[trial_name] = {"path":line}
                else:
                    match = match.groupdict()
                    flag = match[distinguish_client_by]
                    match.pop(distinguish_client_by)
                    update = {f"{flag}_{k}":v for k,v in match.items()}
                    result[trial_name].update(update)            
            except:
                continue

    df = pd.DataFrame(result).T
    if is_save and output_file is not None:
        df.to_csv(output_file, index=False)

    return df

# def parse_log(log_file,
#               output_file=None,
#               client_regex=r"client: (?P<client>\d) round: (?P<round>\d+).+epochs:1.+valid local loss:(?P<val_loss>\d+\.\d+), valid local acc:(?P<val_acc>\d\.\d+).+test local loss:(?P<test_loss>\d+\.\d+), test local acc:(?P<test_acc>0\.\d+)",
#               is_save=False):
#     with open(log_file, "r") as fd:
#         text = fd.read()
#         matches = re.findall(client_regex, text)
#     df = pd.DataFrame(matches, columns=["client", "round", "val_loss", "val_acc", "test_loss", "test_acc"]).astype({"client":int, "round":int, "val_loss":float, "val_acc":float, "test_loss":float, "test_acc":float})
#     if is_save and output_file is not None:
#         df.to_csv(output_file, index=False)

def spec_hyperparam(df:pd.DataFrame, spec_list, path_col, config_file="config.yaml", path_jump=3):
    df["head_path"] = df[path_col].apply(lambda x: "/".join(x.split("/")[:-path_jump]))
    spec_dict = {}
    configs = []
    for head_path in df["head_path"]:
        configs.append(YamlParser(os.path.join(head_path, config_file)))
    for hp_key in spec_list:
        spec_dict[hp_key] = []
        split = hp_key.split(".")
        for config in configs:
            if len(split) == 1:
                spec_dict[hp_key].append(config[hp_key])
            elif len(split) == 2:
                spec_dict[hp_key].append(config[split[0]][split[1]])
            elif len(split) == 3:
                spec_dict[hp_key].append(config[split[0]][split[1]][split[2]])
            elif len(split) == 4:
                spec_dict[hp_key].append(config[split[0]][split[1]][split[2]][split[3]])
    df = pd.concat([df, pd.DataFrame(spec_dict, index=df.index)], axis=1)
    return df