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
