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



class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class WeightedMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        thr1 = torch.log1p(torch.tensor(5.0, device=y_true.device))
        thr2 = torch.log1p(torch.tensor(25.0, device=y_true.device))
        thr3 = torch.log1p(torch.tensor(50.0, device=y_true.device))

        weights = torch.ones_like(y_true)
        weights[(y_true >= thr1) & (y_true < thr2)] = 3.0
        weights[(y_true >= thr2) & (y_true < thr3)] = 5.0
        weights[y_true >= thr3] = 10.0

        loss = weights * torch.abs(y_true - y_pred)
        return torch.sum(loss) / torch.sum(weights)


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        thr1 = torch.log1p(torch.tensor(5.0, device=target.device))
        thr2 = torch.log1p(torch.tensor(25.0, device=target.device))
        thr3 = torch.log1p(torch.tensor(50.0, device=target.device))

        weights = torch.ones_like(target)
        weights[(target >= thr1) & (target < thr2)] = 2.0
        weights[(target >= thr2) & (target < thr3)] = 5.0
        weights[target >= thr3] = 10.0

        loss = weights * (pred - target) ** 2
        return torch.sum(loss) / torch.sum(weights)


class AsymmetricLoss(nn.Module):
    def __init__(self, over_penalty=1.0, under_penalty=5.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(
            diff >= 0, 
            self.over_penalty * diff**2, # over-prediction
            self.under_penalty * diff**2 # under-prediction
        )
        return loss.mean()

class AsymmetricMAELoss(nn.Module):
    def __init__(self, over_penalty=1.0, under_penalty=5.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(
            diff >= 0,
            self.over_penalty * torch.abs(diff), # over-prediction
            self.under_penalty * torch.abs(diff) # under-prediction
        )
        return loss.mean()

class TargetAwareAsymmetricMSELoss(nn.Module):
    def __init__(self, base_over_penalty=1.0, base_under_penalty=5.0):
        super().__init__()
        self.base_over = base_over_penalty
        self.base_under = base_under_penalty

        self.light = torch.log1p(torch.tensor(5.0))
        self.moderate = torch.log1p(torch.tensor(25.0))
        self.heavy = torch.log1p(torch.tensor(50.0))

    def forward(self, pred, target):
        diff = pred - target

        weights = (
            (target < self.light).float() * 1 +
            ((target >= self.light) & (target < self.moderate)).float() * 2 +
            ((target >= self.moderate) & (target < self.heavy)).float() * 5 +
            (target >= self.heavy).float() * 10
        )

        penalty_under = self.base_under * weights

        loss = torch.where(
            diff >= 0,
            self.base_over * diff**2, # over-prediction
            penalty_under * diff**2   # under-prediction
        )
        return loss.mean()

class TargetAwareAsymmetricMAELoss(nn.Module):
    def __init__(self, base_over_penalty=1.0, base_under_penalty=5.0):
        super().__init__()
        self.base_over = base_over_penalty
        self.base_under = base_under_penalty

        self.light = torch.log1p(torch.tensor(5.0))
        self.moderate = torch.log1p(torch.tensor(25.0))
        self.heavy = torch.log1p(torch.tensor(50.0))

    def forward(self, pred, target):
        diff = pred - target

        weights = (
            (target < self.light).float() * 1 +
            ((target >= self.light) & (target < self.moderate)).float() * 2 +
            ((target >= self.moderate) & (target < self.heavy)).float() * 5 +
            (target >= self.heavy).float() * 10
        )

        penalty_under = self.base_under * weights

        loss = torch.where(
            diff >= 0,
            self.base_over * diff.abs(), # over-prediction
            penalty_under * diff.abs()   # under-prediction
        )
        return loss.mean()
