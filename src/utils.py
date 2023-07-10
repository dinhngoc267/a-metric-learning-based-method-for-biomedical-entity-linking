import torch 
import numpy as np
import torch.nn as nn


def pairwise_euclidean_dist(a:torch.Tensor, b:torch.Tensor):
    pdist = nn.PairwiseDistance(p=2)

    output = pdist(a, b)
    return output

def euclidean_dist(a: torch.Tensor, b:torch.Tensor):
    if len(a.shape) == 1:
        a = a.view((1,)+a.shape)
    if len(b.shape) == 1:
        b = b.view((1,)+b.shape)
    dist_matrix = torch.cdist(a, b, p=2)    
    return dist_matrix