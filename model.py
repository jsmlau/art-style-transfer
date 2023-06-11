from typing import Dict

import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19(nn.Module):
    def __init__(self, layers: Dict[str, str] = None):
        super().__init__()
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1',
                             'conv5_1']

        if not layers:
            self.layers = {
                '0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'
            }

        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = self.features[:max(self.layers.keys()) + 1]
        
        # Freeze all VGG parameters since we're only optimizing the target image
        for param in self.features.parameters():
            param.requires_grad = False

        # Replace all max-pooling layers in the network with average pooling for better-looking results
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                self.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Pass an image forward through the VGG19 network to get the features for a set of layers.
        
        Returns:

        """
        features = {}

        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x

        return features


if __name__ == '__main__':
    vgg = VGG19()
