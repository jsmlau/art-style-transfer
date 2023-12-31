from typing import List, Dict

import torch
import torch.nn as nn
from torch import Tensor


class StyleLoss(nn.Module):
    def __init__(self, device: str='cpu') -> None:
        super().__init__()
        self.device = device

    def gram_matrix(self, features: Tensor) -> Tensor:
        """
        Compute the Gram matrix from cnn.

        Args:
            features: A PyTorch Variable of shape (N, C, H, W) representing the cnn for a batch of N images.
            normalize: An optional boolean indicating whether to normalize the Gram matrix. If set to True, the Gram matrix will be divided by the number of neurons (H * W * C).

        Returns: A PyTorch Variable of shape (N, C, C) providing the Gram matrices for the N input images, optionally
        normalized.

        """
        N, C, H, W = features.size()

        # F_ik: unrolled intermediate representation
        # F_jk: transpose of F_ik

        F_ik = features.view(N, C, H * W)  # feature map
        F_jk = torch.transpose(input=F_ik, dim0=1, dim1=2)
        G = torch.bmm(F_ik, F_jk)  # Gram Matrix

        return G.div(H * W * C).to(self.device)

    def forward(self,
                style_weight: float,
                style_weight_ratio: Dict[str, float],
                features: Dict[str, Tensor],
                style_targets: Dict[str, Tensor],
                style_layers: List[str]) -> float:
        """
        Compute the style loss for a set of layers.

        Args:
            style_weight: The weight assigned to the style loss.
            style_weight_ratio: A list of ratio of weight to be assigned to each layer in the style_layers.
            features: A list of cnn at each layer of the current image, generated by the extract_features function.
            style_layers: A list of indices corresponding to the layers included in the style loss calculation.
            style_targets: A list of PyTorch Variables representing the Gram matrices computed from the source style image at each layer specified in style_layers.


        Returns: A PyTorch Variable holding a scalar value representing the computed style loss.

        """
        style_loss = 0.0

        for i, layer in enumerate(style_layers):
            # compute gram matrix
            feature = features[layer]
            N, C, H, W = feature.size()
            gram_layer = self.gram_matrix(features=feature)
            layer_style_loss = torch.sum((gram_layer - style_targets[layer])**2)
            style_loss += (layer_style_loss / (gram_layer - style_targets[layer]).abs().sum().add(1e-8) * style_weight_ratio[layer])

        return style_loss * style_weight
        
