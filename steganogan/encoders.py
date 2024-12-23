# -*- coding: utf-8 -*-

import torch
from torch import nn


class BasicEncoder(nn.Module):
    def __init__(self, data_depth, hidden_size, device):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.device = device  # Store device
        self._models = self._build_models(device)

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self, device):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)
        
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
            nn.Tanh(),
        ).to(device)

        return self.features, self.layers

    def forward(self, image, data):
        image = image.to(self.device)  # Ensure the image is on the correct device
        data = data.to(self.device)  # Ensure the data is on the correct device

        # Reshape and expand `data` to match spatial dimensions of `image`
        batch_size, data_depth = data.size()
        height, width = image.size(2), image.size(3)
        data = data.view(batch_size, data_depth, 1, 1).expand(batch_size, data_depth, height, width)

        # Pass `image` through feature extractor
        x = self.features(image)

        # Concatenate the image features with the data
        x = torch.cat([x, data], dim=1)

        # Pass the combined tensor through the remaining layers
        x = self.layers(x)

        if self.add_image:
            x = image + x  # Add the cover image to the output for residual behavior

        return x





class ResidualEncoder(BasicEncoder):
    """
    The ResidualEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image.
    """

    add_image = True

    def _build_models(self, device):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)
        
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
        ).to(device)

        return self.features, self.layers

    def __init__(self, data_depth, hidden_size, device):
        super().__init__(data_depth, hidden_size, device)


class DenseEncoder(BasicEncoder):
    """
    The DenseEncoder module takes a cover image and a data tensor and combines
    them into a steganographic image using dense connections.
    """

    add_image = True  # Indicates whether to add the cover image to the output

    def _build_models(self, device):
        # Define the layers with dense connections
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        ).to(device)

        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3),
            nn.Tanh(),  # Output values in the range [-1, 1]
        ).to(device)

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image, data):
        image = image.to(self.device)  # Ensure the image is on the correct device
        data = data.to(self.device)  # Ensure the data is on the correct device

        # Reshape and expand `data` to match spatial dimensions of `image`
        batch_size, data_depth = data.size()
        height, width = image.size(2), image.size(3)
        data = data.view(batch_size, data_depth, 1, 1).expand(batch_size, data_depth, height, width)

        # Pass the image through the first convolutional layer
        x1 = self.conv1(image)

        # Concatenate `x1` with `data` and pass through the second layer
        x2_input = torch.cat([x1, data], dim=1)
        print(f"x2_input shape: {x2_input.shape}")
        x2 = self.conv2(x2_input)
        


        # Concatenate `x1`, `x2`, and `data`, then pass through the third layer
        x3_input = torch.cat([x1, x2, data], dim=1)
        x3 = self.conv3(x3_input)

        # Concatenate all previous outputs and `data`, then pass through the final layer
        x4_input = torch.cat([x1, x2, x3, data], dim=1)
        x4 = self.conv4(x4_input)

        if self.add_image:
            x4 = image + x4  # Add the cover image to the output for residual behavior

        return x4
