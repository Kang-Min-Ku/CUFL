# CUFL Running Guide

We recommend creating separate conda environments for **data generation** and code **execution**.

## Data Generation

### Requirement

1. Python Version: 3.8.18
2. Module
   - pytorch 2.0.0
   - torch-geometric 2.0.4
   - ogb 1.3.6
   - networkx 3.1
   - metis-python 0.1, [Link](https://github.com/james77777778/metis_python)

### How to Run

```console
cd federated/data/generator
python3 split_disjoint.py
```

## Execution

### Requirement

1. Python Version: 3.10.14
2. Module
   - pytorch 2.1.1
   - torch-geometric 2.5.3
   - networkx 3.2.1
   - numpy 1.26.3
   - scipy 1.14.0
   - pyyaml 6.0.1

### Config file

All variables necessary for execution, such as paths and hyperparameters, are stored in **YAML files** in config directory

### How to Run

Pretrain

```console
python3 main.py [pretrain config file path]
```

Run CUFL

```console
python3 main.py [config file path]
```

Here is an example

```console
python3 main.py --config config/fedprox_pretrain.yaml
python3 main.py --config config/cufl_cora.yaml
```

# Hyperparameters setting
```
GNN hidden dim : 128

```
