import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, use_sigmoid=False, square=True):
        super(DiceLoss, self).__init__()
        
        self.smooth = smooth
        self.use_sigmoid = use_sigmoid
        self.square = square

    def forward(self, prediction, target):

        if target.dtype is not torch.float:
            target = target.float()

        # Get the batch size
        N = prediction.shape[0]

        # Prediction
        if self.use_sigmoid:
            prediction = prediction.sigmoid()
            prediction = prediction[:,0]
        else:
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction[:,1]
        
        # Flatten
        prediction_flat = prediction.view(N, -1)
        target_flat = target.view(N, -1)

        # Dice Loss
        intersection = prediction_flat * target_flat
        molecular = 2.0 * intersection.sum(dim=1) + self.smooth
        if self.square:
            prediction_flat = prediction_flat.pow(2)
        denominator = prediction_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth

        loss = molecular / denominator
        loss = 1.0 - loss.sum() / N

        return loss