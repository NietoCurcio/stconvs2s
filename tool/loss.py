import torch
import torch.nn as nn
import numpy as np
# from sklearn.utils.class_weight import compute_sample_weight

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
    
class EPLLossFirstLeadTime(nn.Module):
    def __init__(self, n_out=5, gamma=3.258096538021482):
        """
        n_out: int, the number of output timesteps.
        gamma: float, threshold for identifying extreme events.
        """
        super(EPLLossFirstLeadTime, self).__init__()
        self.n_out = n_out
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        """
        Computes the EPL loss based on the first timestep prediction.
        
        y_pred: Tensor of shape (batch_size, channels, timesteps, height, width)
        y_true: Tensor of shape (batch_size, channels, timesteps, height, width)
        """
        # Select the first timestep (index 0) for t+1 prediction
        y_pred_first = y_pred[:, :, 0, :, :]
        y_true_first = y_true[:, :, 0, :, :]

        # Compute the error for the first timestep
        c = y_pred_first - y_true_first
        b = y_true_first

        # For normal events (b < gamma), use squared error loss
        normal_mask = b < self.gamma
        loss_normal = c[normal_mask] ** 2

        # For extreme events (b >= gamma), apply the exponential penalty
        extreme_mask = ~normal_mask
        c_ext = c[extreme_mask]
        loss_ext = torch.where(
            c_ext >= 0,
            torch.exp(c_ext / (self.n_out + 1)) - 1,
            torch.exp(-c_ext) - 1
        )

        # Combine losses from normal and extreme cases
        if loss_normal.numel() + loss_ext.numel() > 0:
            total_loss = torch.cat([loss_normal.view(-1), loss_ext.view(-1)])
            return total_loss.mean()
        else:
            return torch.tensor(0.0, device=y_pred.device)

class MultiClassTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, class_weights=None, k=10.0):
        """
        Multi-class Tversky Loss for imbalanced precipitation prediction.

        Classes (in log-space):
            0: weak     < log1p(5)
            1: moderate [log1p(5), log1p(25))
            2: heavy    [log1p(25), log1p(50))
            3: extreme  >= log1p(50)

        Parameters:
            alpha (float): Weight for false positives.
            beta (float): Weight for false negatives.
            smooth (float): Smoothing term.
            class_weights (list or tensor): Weights for each class.
            k (float): Steepness parameter for the soft one-hot encoding.
        """
        super(MultiClassTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.k = k
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        else:
            self.class_weights = torch.tensor(class_weights)

    def forward(self, y_pred, y_true):
        """
        y_pred and y_true are assumed to be in log-space and have the same shape.
        y_true is hard-encoded and y_pred is soft-encoded.
        """
        num_classes = 4
        one_hot_true = self._hard_one_hot_encode(y_true, num_classes)
        one_hot_pred = self._soft_one_hot_encode(y_pred, num_classes)
        
        total_loss = 0.0
        total_weight = 0.0
        for c in range(num_classes):
            true_c = one_hot_true[:, c]
            pred_c = one_hot_pred[:, c]
            true_c_flat = true_c.contiguous().view(-1)
            pred_c_flat = pred_c.contiguous().view(-1)
            
            TP = (pred_c_flat * true_c_flat).sum()
            FP = (pred_c_flat * (1 - true_c_flat)).sum()
            FN = ((1 - pred_c_flat) * true_c_flat).sum()
            
            tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            loss_c = 1 - tversky_index
            weight = self.class_weights[c]
            total_loss += weight * loss_c
            total_weight += weight
        
        return total_loss / total_weight

    def _hard_one_hot_encode(self, tensor, num_classes):
        shape = list(tensor.shape)
        one_hot = torch.zeros([shape[0], num_classes] + shape[1:], device=tensor.device)
        
        moderate = torch.log1p(torch.tensor(5.0, device=tensor.device)).item()
        heavy = torch.log1p(torch.tensor(25.0, device=tensor.device)).item()
        extreme = torch.log1p(torch.tensor(50.0, device=tensor.device)).item()
        
        mask0 = tensor < moderate
        mask1 = (tensor >= moderate) & (tensor < heavy)
        mask2 = (tensor >= heavy) & (tensor < extreme)
        mask3 = tensor >= extreme
        
        one_hot[:, 0][mask0] = 1.0
        one_hot[:, 1][mask1] = 1.0
        one_hot[:, 2][mask2] = 1.0
        one_hot[:, 3][mask3] = 1.0
        
        return one_hot

    def _soft_one_hot_encode(self, tensor, num_classes):
        # Define thresholds in log-space
        moderate = torch.log1p(torch.tensor(5.0, device=tensor.device)).item()
        heavy = torch.log1p(torch.tensor(25.0, device=tensor.device)).item()
        extreme = torch.log1p(torch.tensor(50.0, device=tensor.device)).item()
        
        # Use sigmoids to get soft probabilities
        p0 = 1 - torch.sigmoid(self.k * (tensor - moderate))
        p1 = torch.sigmoid(self.k * (tensor - moderate)) - torch.sigmoid(self.k * (tensor - heavy))
        p2 = torch.sigmoid(self.k * (tensor - heavy)) - torch.sigmoid(self.k * (tensor - extreme))
        p3 = torch.sigmoid(self.k * (tensor - extreme))
        
        one_hot_soft = torch.stack([p0, p1, p2, p3], dim=1)
        one_hot_soft = one_hot_soft / (one_hot_soft.sum(dim=1, keepdim=True) + 1e-8)
        return one_hot_soft
    