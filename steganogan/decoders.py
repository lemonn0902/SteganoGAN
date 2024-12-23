# -*- coding: utf-8 -*-

import torch
from torch import nn


class BasicDecoder(nn.Module):
    """
    Decodes a steganographic image to retrieve the embedded data tensor.

    Input: (N, 3, H, W) - Steganographic image
    Output: (N, D, H, W) - Decoded data tensor
    """

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_models(self):
        return nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.data_depth)
        )

    def __init__(self, data_depth, hidden_size, device="cpu"):
        super().__init__()
        self.version = '1'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.device = device
        self.layers = self._build_models().to(device)

    def forward(self, x):
        # Ensure tensor has the expected shape (N, 3, H, W)
        if len(x.shape) != 4:
            raise ValueError(f"Expected input shape (N, 3, H, W), but got {x.shape}")

        x = x.to(self.device)  # Move input to the correct device
        return self.layers(x)


class DenseDecoder(BasicDecoder):
    """
    Decodes a steganographic image using dense connections to retrieve the embedded data tensor.

    Input: (N, 3, H, W) - Steganographic image
    Output: (N, D, H, W) - Decoded data tensor
    """

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        ).to(self.device)

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        ).to(self.device)

        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        ).to(self.device)

        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth),
            nn.Tanh()  # Ensure output is in the range [-1, 1]
        ).to(self.device)

        return nn.ModuleList([self.conv1, self.conv2, self.conv3, self.conv4])

    def forward(self, x):
        # Ensure tensor has the expected shape (N, 3, H, W)
        if len(x.shape) != 4:
            raise ValueError(f"Expected input shape (N, 3, H, W), but got {x.shape}")

        x = x.to(self.device)  # Move input to the correct device
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))

        return x4
