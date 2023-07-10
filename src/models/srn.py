import torch.nn as nn
import torch
import torch.nn.functional as F

class Radius(nn.Module):
    def __init__(self, init_value = 10.0, epsilon = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.radius =  nn.Parameter(torch.tensor(init_value))
        
    def forward(self):
        return self.radius
    
    def loss(self, y_true):
        return F.mse_loss(self.radius, y_true)
    