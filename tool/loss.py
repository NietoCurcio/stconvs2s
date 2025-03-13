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
    
class EPL(nn.Module):
    """
    Extreme Penalized Loss (EPL) for imbalanced time series prediction.
    
    Loss definition:
        f(x) = { 
                  x^2,                             if y_true < gamma, 
                  exp(-x) - 1,                     if y_true >= gamma and x < 0,
                  exp(x / lambda_) - 1,              if y_true >= gamma and x >= 0,
                }
    where:
      - x = y_pred - y_true,
      - gamma is the threshold to decide extreme events,
      - lambda_ controls the penalty strength for over-predicted extremes.
    
    Args:
        gamma (float): Threshold value to decide extreme events.
        lambda_ (float): Scaling parameter for over-predicted extreme events.
                         Typically set to n_out + 1.
        reduction (str): Specifies the reduction: 'mean' | 'sum' | 'none'.
    """
    def __init__(self, gamma=None, lambda_=6.0, reduction='mean'):
        super(EPL, self).__init__()
        self.gamma = gamma  # may be set later using set_gamma method
        self.lambda_ = lambda_
        self.reduction = reduction
        print(f"EP Loss: gamma={gamma}, lambda_={lambda_}, reduction={reduction}")

    def forward(self, y_pred, y_true):
        if self.gamma is None:
            raise ValueError("Gamma is not set. Please call set_gamma() before using the loss.")
            
        # Compute the error (difference between prediction and true value)
        error = y_pred - y_true
        
        # For normal events: when ground-truth is below gamma, use squared error.
        normal_loss = error ** 2
        
        # Extreme events: where ground-truth is >= gamma
        extreme_mask = (y_true >= self.gamma)
        
        # Initialize loss tensor for extreme events.
        extreme_loss = torch.zeros_like(error)
        
        # Under-prediction: error < 0
        under_mask = extreme_mask & (error < 0)
        # Over-prediction: error >= 0
        over_mask = extreme_mask & (error >= 0)
        
        # Apply exponential penalty for extreme events.
        extreme_loss[under_mask] = torch.exp(-error[under_mask]) - 1
        extreme_loss[over_mask] = torch.exp(error[over_mask] / self.lambda_) - 1
        
        # Combine losses: use extreme_loss for extreme events, normal_loss otherwise.
        loss = torch.where(extreme_mask, extreme_loss, normal_loss)
        
        # Apply reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def set_gamma(self, bin_size, data):
        """
        Computes gamma from the provided data using a histogram-based method.
        
        Args:
            bin_size (float): The size of each bin in the histogram.
            data (array-like): The dataset (assumed to be 1D and continuous).
        
        This method sets self.gamma based on the following logic:
          - It creates a histogram of the data.
          - It calculates the ratio of counts in each bin.
          - It selects the first bin where the ratio is less than 1/num_bins,
            the previous bin ratio is >= 1/num_bins, and (if possible) the next
            bin ratio is also below 1/num_bins.
        """
        data = np.asarray(data)
        min_val = data.min()
        max_val = data.max()
        bins_num = int((max_val - min_val) / bin_size) + 1
        
        counts, bin_edges = np.histogram(data, bins=bins_num)
        bin_ratios = counts / counts.sum()
        gamma = None
        
        # Iterate from 1 to bins_num - 1 to safely access i+1.
        for i in range(1, bins_num - 1):
            if bin_ratios[i] < 1 / bins_num and bin_ratios[i - 1] >= 1 / bins_num and bin_ratios[i + 1] < 1 / bins_num:
                gamma = bin_edges[i]
                break
                
        if gamma is None:
            raise ValueError("Unable to determine gamma from the data using the given bin_size.")
            
        self.gamma = round(gamma, 1)
        print(f"EPL Gamma set to {self.gamma} (before log transformation)")
        self.gamma = np.log1p(self.gamma)
        print(f"EPL Gamma set to {self.gamma}")
        return self.gamma
