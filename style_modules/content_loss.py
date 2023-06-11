import torch
import torch.nn as nn
from torch import Tensor


class ContentLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, content_weight: float, content_current: Tensor,
                content_target: Tensor) -> Tensor:
        """
        Compute the content loss for style transfer.
        The content loss is calculated using the following equation: L_c = w_c * (sum_ij(F_ij - P_ij) ^2) / 2
        
        Args:
            content_weight: A scalar value representing the weight of the content loss.
            content_current: A Pytorch tensor of shape (1, C_l, H_l, W_l) representing the cnn of the current image.
            content_target: A Pytorch tensor of shape (1, C_l, H_l, W_l) representing the cnn of the original image.

        Returns: A scalar tensor representing the computed content loss.

        """
        content_loss = torch.sum((content_current - content_target) ** 2)
        content_loss *= content_weight
        return content_loss
