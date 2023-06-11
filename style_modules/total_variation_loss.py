import torch
import torch.nn as nn
from torch.autograd import Variable


class TotalVariationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: Variable, total_variation_weight: float=2e-2) -> Variable:
        """
        Compute total variation loss.

        Args:
            img: A PyTorch Variable of shape (1, 3, H, W) representing an input image.
            total_variation_weight: A scalar representing the weight (w_t) to be used for the TV loss.

        Returns: A PyTorch Variable representing a scalar that gives the total variation loss for the input image, weighted by
        total_variation_weight.
        """

        W_tv = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        H_tv = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

        loss_tv = W_tv + H_tv

        return loss_tv * total_variation_weight
