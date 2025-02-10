import torch
from ..utils.util import *
from ..utils.torchutil import torch_load

class DataLoader:
    def __init__(self, args, is_server=False, num_workers=1):
        self.args = args
        self.num_workers = num_workers
        self.is_server = is_server
        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader
        self.client_id = -1
    
    def get_data(self, mode, client_id=-1):
        if mode in ["test", "valid"]:
            data = torch_load(
                self.args.data_path,
                f"{self.args.task}/{self.args.num_clients}/{mode}.pt"
            )["data"]
        else:
            data = torch_load(
                self.args.data_path,
                f"{self.args.task}/{self.args.num_clients}/{self.args.data_header}_{client_id}.pt"
            )["client_data"]

        return [data]
    
    def get_loader(self, client_id):
        if self.is_server and self.args.eval_global:
            self.test = self.get_data(mode='test')
            self.test_loader = self.DataLoader(
                dataset=self.test,
                batch_size=1, 
                shuffle=False, num_workers=self.num_workers, pin_memory=False) #Should it be self.DataLoader?
            self.valid = self.get_data(mode='valid')
            self.valid_loader = self.DataLoader(
                dataset=self.valid,
                batch_size=1, 
                shuffle=False, num_workers=self.num_workers, pin_memory=False)
        elif not self.is_server and self.client_id != client_id:
            self.client_id = client_id
            self.partition = self.get_data(mode="partition", client_id=client_id)
            self.train = self.get_data(mode='train', client_id=client_id)
            self.train_loader = self.DataLoader(
                dataset=self.train,
                batch_size=1, 
                shuffle=False, num_workers=self.num_workers, pin_memory=False)

            if self.args.eval_global:
                self.test = self.get_data(mode='test', client_id=client_id)
                self.test_loader = self.DataLoader(
                    dataset=self.test,
                    batch_size=1, 
                    shuffle=False, num_workers=self.num_workers, pin_memory=False)
                self.valid = self.get_data(mode='valid', client_id=client_id)
                self.valid_loader = self.DataLoader(
                    dataset=self.valid,
                    batch_size=1, 
                    shuffle=False, num_workers=self.num_workers, pin_memory=False)