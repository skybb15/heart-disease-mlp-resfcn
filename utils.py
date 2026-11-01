# utils.py
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, input_dim, path):
    model = model_class(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
