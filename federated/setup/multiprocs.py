import os
import sys
import time
import torch
import random
import atexit
import numpy as np
import multiprocessing #as mp
import torch.multiprocessing as mp

from datetime import datetime
from ..utils.util import print_and_log

class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = args.gpu #[int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0]
        self.proc_id = os.getppid()
        self.logger = args.logger

        msg = f'main process id: {self.proc_id}'
        print_and_log(msg, self.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)
        
        self.sd = mp.Manager().dict()
        self.sd['is_done'] = False
        self.create_workers(Client)
        self.server = Server(args, self.sd, self.gpu_server)
        atexit.register(self.done)

        if self.args.tuning_analysis:
            os.makedirs(self.args.tuning_config["analyze_result_path"], exist_ok=True)

    def create_workers(self, Client):
        self.processes = []
        self.q = {}
        for worker_id in range(self.args.num_clients):
            gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            
            msg = f'worker_id: {worker_id}, gpu_id:{gpu_id}'
            print_and_log(msg, self.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)
            
            self.q[worker_id] = mp.Queue()
            p = mp.Process(target=WorkerProcess, args=(self.args, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            p.start()
            self.processes.append(p)

    def start(self):
        self.sd['is_done'] = False
        if os.path.isdir(self.args.checkpoint_path) == False:
            os.makedirs(self.args.checkpoint_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.num_clients*self.args.fraction)
        for curr_round in range(self.args.num_rounds):
            self.curr_round = curr_round
            self.updated = set()
            np.random.seed(self.args.seed+curr_round)
            self.selected = np.random.choice(self.args.num_clients, self.n_connected, replace=False).tolist()

            st = time.time()
            ##################################################
            self.server.on_round_begin(self.selected, curr_round)
            ##################################################
            while len(self.selected)>0:
                _selected = []
                for worker_id, q in self.q.items():
                    c_id = self.selected.pop(0)
                    _selected.append(c_id)
                    q.put((c_id, curr_round))
                    if len(self.selected) == 0:
                        break
                self.wait(curr_round, _selected)
            # print(f'[main] all clients updated at round {curr_round}')
            ###########################################
            self.server.on_round_complete(self.updated)
            ###########################################
            msg = f'[main] round {curr_round+1} done ({time.time()-st:.2f} s)'
            print_and_log(msg, self.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)

        self.sd['is_done'] = True
        for worker_id, q in self.q.items():
            q.put(None)

        msg = '[main] server done'
        print_and_log(msg, self.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)
        sys.exit()

    def wait(self, curr_round, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if not c_id in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(0.1)

    def done(self):
        for p in self.processes:
            p.join()
        
        msg = "[main] All children have joined. Destroying main process ..."
        print_and_log(msg, self.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)
            

class WorkerProcess:
    def __init__(self, args, worker_id, gpu_id, q, sd, Client):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False
        self.client = Client(self.args, self.worker_id, self.gpu_id, self.sd)
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            mesg = self.q.get()
            if not mesg == None:
                client_id, curr_round = mesg 
                ##################################
                self.client.switch_state(client_id)
                # self.client.on_receive_message(curr_round)
                # self.client.on_round_begin(curr_round)
                self.client.local_job(curr_round)
                self.client.save_state()
                ##################################
            time.sleep(1.0)

        msg = '[main] Terminating worker processes ... '
        print_and_log(msg, self.client.logger, self.args.verbose_print_setup, self.args.verbose_log_setup)
        sys.exit()





