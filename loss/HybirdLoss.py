import torch
import torch.nn as nn
import torch.nn.functional as F

from .DiceLoss import DiceLoss

class HybirdLoss(nn.Module):
    """ 
        Hybrid Loss = Dice Loss + Cross Entropy Loss 
        input : prediction with shape (N, C, d1, d2, ...) or (N, d1, d2, ...)
                target with shape (N, d1, d2, ...)
    """

    def __init__(self, weight=None, with_channel=True):
        super(HybirdLoss, self).__init__()
        
        if weight is None:
            self.weight = nn.Parameter(torch.Tensor([1.0, 1.0]), requires_grad=False)
        else:
            self.weight = nn.Parameter(weight, requires_grad=False)
        self.with_channel = with_channel
        
        if self.with_channel:
            self.dice = DiceLoss()
            self.bce = nn.CrossEntropyLoss()
        else:
            self.dice = DiceLoss(use_sigmoid=True)
            self.bce = nn.BCELoss()

    def forward(self, prediction, target):

        if self.with_channel:
            loss = self.weight[0] * self.dice(prediction, target) + \
                self.weight[1] * self.bce(prediction, target)
        else:
            loss = self.weight[0] * self.dice(prediction, target) + \
                self.weight[1] * self.bce(prediction.sigmoid(), target)

        return loss