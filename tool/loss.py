import torch
import torch.nn as nn

class TweedieLoss(nn.Module):
    def __init__(self, variance_power, eps=1e-6):
        """
        Initializes the Tweedie loss.
        
        Args:
            variance_power (float): The Tweedie power parameter p (should be in (1,2)).
            eps (float): A small constant for numerical stability.
        """
        super().__init__()
        self.variance_power = variance_power
        self.eps = eps
        
    def forward(self, yhat, y, weights=None):
        """
        Compute the Tweedie loss.
        
        Args:
            yhat (torch.Tensor): Predictions on the log-scale (i.e., F(x) where mu=exp(F(x))).
            y (torch.Tensor): True target values.
            weights (torch.Tensor, optional): Optional weights for each observation.
            
        Returns:
            torch.Tensor: The computed Tweedie loss (averaged over observations).
        """
        p = self.variance_power
        
        # For stability, we might want to clamp the predictions.
        yhat = torch.clamp(yhat, min=-50, max=50)
        
        # Convert log-scale predictions to the mean scale
        # mu = exp(F(x))
        # Compute each term of the loss
        term1 = torch.exp((2 - p) * yhat) / (2 - p)
        term2 = y * torch.exp((1 - p) * yhat) / (1 - p)
        
        loss = term1 - term2
        
        # Optionally, apply observation weights if provided

        weights = torch.ones_like(y)
        mask_common   = (y >= 0)   & (y < 5)
        mask_second   = (y >= 5)   & (y < 25)
        mask_third    = (y >= 25)  & (y < 50)
        mask_extreme  = (y >= 50)  & (y < 150)

        weights[mask_common]  = 1.0
        weights[mask_second]  = 10.0
        weights[mask_third]   = 50.0
        weights[mask_extreme] = 100.0

        if weights is not None:
            loss = loss * weights
        
        # Return the mean loss over all observations
        return torch.mean(loss)
