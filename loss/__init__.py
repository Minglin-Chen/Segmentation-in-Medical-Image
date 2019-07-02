import torch
import torch.nn as nn
import torch.nn.functional as F

from .DiceLoss import DiceLoss
from .HybirdLoss import HybirdLoss

crtierion = {
    'dice': DiceLoss,
    'ce': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
    'hybrid': HybirdLoss
}

def criterion_provider(name, **kwargs):

    loss = crtierion[name](**kwargs)
    
    return loss