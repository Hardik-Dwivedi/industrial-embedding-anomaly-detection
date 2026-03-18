import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience, min_delta, path = '../working/checkpoint.pt', verbose = True):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'Early Stopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Metric Improved: Saving model to {self.path}...')
        torch.save(model.state_dict(), self.path)
