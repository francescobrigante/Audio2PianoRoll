# ==============================================================================
# Training loss: weighted combination of BCE (note on/off) and MSE
# (activation intensity) over the predicted piano roll.
# ==============================================================================

import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combines BCE and MSE loss.
        alpha: Weighting factor for BCE (0.5 means equal importance for BCE and MSE).
        """
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        mse_loss = self.mse(output, target)
        return self.alpha * bce_loss + (1 - self.alpha) * mse_loss
