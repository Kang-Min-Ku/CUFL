import numpy as np
from collections import deque

class GSS:
    def __init__(self, init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger=False):
        self.scale = init_scale
        self.window_size = window_size
        self.patience = patience
        self.varying_factor = varying_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.prefer_larger = prefer_larger

        self.accum_metric = deque(maxlen=window_size)
        self.best_metric = np.inf
        self.num_good = 0
        self.num_bad = 0
        self.cache = None

    def compare(self, x, y):
        # x: challenger y: best metric
        if self.prefer_larger:
            return x > y
        else:
            return x < y

    def improve(self, flag):
        if flag:
            self.scale *= self.varying_factor
        else:
            self.scale /= self.varying_factor
    
    def clip(self):
        if self.scale > self.max_scale:
            self.scale = self.max_scale
        elif self.scale < self.min_scale:
            self.scale = self.min_scale

    def save_score(self, score):
        self.cache = score

class UnitGSS(GSS):
    #Unit Greedy Scale Scheduler
    def __init__(self, init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger=False):
        super().__init__(init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger)

    def evaluate(self, val_metric):
        self.accum_metric.append(val_metric)

        #if val_metric < self.best_metric:
        if self.compare(np.mean(self.accum_metric), self.best_metric):
            self.best_metric = np.mean(self.accum_metric)
            self.num_good += 1
            self.num_bad = 0
        else:
            self.num_good = 0
            self.num_bad += 1
        
        if self.num_good >= self.patience:
            self.improve(True)
            self.num_good = 0
        elif self.num_bad >= self.patience:
            self.improve(False)
            self.num_bad = 0

        self.clip()

class DiffGSS(GSS):
    #Difference Greedy Scale Scheduler
    def __init__(self, init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger=False):
        super().__init__(init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger)
    
    def evaluate(self, val_metric):
        if self.cache is None: return
        val_metric = val_metric - self.cache
        self.cache = None

        self.accum_metric.append(val_metric)

        #if val_metric < self.best_metric:
        if self.compare(np.mean(self.accum_metric), self.best_metric):
            self.best_metric = np.mean(self.accum_metric)
            self.num_good += 1
            self.num_bad = 0
        else:
            self.num_good = 0
            self.num_bad += 1
        
        if self.num_good >= self.patience:
            self.improve(True)
            self.num_good = 0
        elif self.num_bad >= self.patience:
            self.improve(False)
            self.num_bad = 0

        self.clip()

class VectorGSS(GSS):
    # Vector Greedy Scale Scheduler
    # If metric gets better, change scale. Otherwise, change direction
    def __init__(self, init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger=False):
        super().__init__(init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger)
        self.is_increase = True # True: increase, False: decrease

    def evaluate(self, val_metric):
        self.accum_metric.append(val_metric)

        #if val_metric < self.best_metric:
        if self.compare(np.mean(self.accum_metric), self.best_metric):
            self.best_metric = np.mean(self.accum_metric)
            self.num_good += 1
            self.num_bad = 0
        else:
            self.num_good = 0
            self.num_bad += 1
        
        if self.num_good >= self.patience:
            self.improve(self.is_increase)
            self.num_good = 0
        elif self.num_bad >= self.patience:
            self.is_increase = not self.is_increase
            self.improve(self.is_increase)
            self.num_bad = 0

        self.clip()

class EMAGSS(GSS):
    # Exponential Moving Average Greedy Scale Scheduler
    # If metric gets better, change scale. Otherwise, change direction
    # Grant more weight on recent metrics
    NotImplementedError