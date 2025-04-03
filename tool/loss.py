import torch
import torch.nn as nn
import numpy as np
# from sklearn.utils.class_weight import compute_sample_weight

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, yhat, y):
        loss = self.loss_fn(yhat, y.float())
        return loss

class BCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCELoss(*args, **kwargs)

    def forward(self, yhat, y):
        loss = self.loss_fn(yhat, y.float())
        return loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()

        y_pred_sum = (y_pred * y_pred).sum()
        y_true_sum = (y_true * y_true).sum()

        dice_coeff = (2.0 * intersection + self.eps) / (y_pred_sum + y_true_sum + self.eps)

        loss = 1 - dice_coeff
        return loss

def compute_weights(y_true):
    y_np = y_true.cpu().detach().numpy().squeeze()
    labels = np.empty_like(y_np, dtype=int)
    thr1 = np.log1p(5)
    thr2 = np.log1p(25)
    thr3 = np.log1p(50)
    labels[y_np < thr1] = 1
    labels[(y_np >= thr1) & (y_np < thr2)] = 2
    labels[(y_np >= thr2) & (y_np < thr3)] = 3
    labels[y_np >= thr3] = 4
    # Extreme: 0.0040%, Heavy: 0.0173%, Moderate: 0.7135%, Weak: 99.2653%
    fixed_weights = {
        1: 1.0,  
        2: 10.0,
        3: 50.0,
        4: 100.0
    }
    sample_weights = np.empty_like(labels, dtype=float)
    for label, weight in fixed_weights.items():
        sample_weights[labels == label] = weight
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32, device=y_true.device)
    return sample_weights.unsqueeze(1)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        weights = compute_weights(y_true)
        # loss = torch.mean(torch.abs(y_true - y_pred))
        # print(f"y_pred.shape: {y_pred.shape}")
        # print(f"y_true.shape: {y_true.shape}")
        # print(f"weights.shape: {weights.shape}")
        # exit(0)
        # return loss
        return torch.sum(weights * torch.abs(y_true - y_pred)) / torch.sum(weights)


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[(target >= 5) & (target < 25)] = 2.0
        weights[(target >= 25) & (target < 50)] = 5.0
        weights[target >= 50] = 10.0

        return torch.mean(weights * (pred - target)**2)

class AsymmetricLoss(nn.Module):
    def __init__(self, over_penalty=1.0, under_penalty=5.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(
            diff >= 0, 
            self.over_penalty * diff**2,  # over-prediction
            self.under_penalty * diff**2  # under-prediction
        )
        return loss.mean()
