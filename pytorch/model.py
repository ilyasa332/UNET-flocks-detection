import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_


class UNet(nn.Module):
    def __init__(self, input_channels: int = 9, depth: int = 6, num_filters: int = 8):
        super().__init__()
        self.downsampling = nn.ModuleList()
        for _ in range(depth):
            self.downsampling.append(nn.Sequential(
                nn.Conv2d(input_channels, num_filters, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding="same"),
                nn.ReLU(),
            ))
            input_channels = num_filters
            num_filters *= 2
        ff2 = 64

        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters // 2, num_filters, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, ff2, kernel_size=2, stride=(2, 2))
        )

        self.upsampling = nn.ModuleList()
        for i in range(depth - 1):
            ff2 = ff2 // 2
            num_filters = num_filters // 2
            self.upsampling.append(
                nn.Sequential(
                    nn.Conv2d(ff2 * 2 + self.downsampling[-i - 1][2].out_channels, num_filters, kernel_size=3,
                              padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding="same"),
                    nn.ReLU(),
                    nn.ConvTranspose2d(num_filters, ff2, kernel_size=2, stride=(2, 2))
                )
            )

        self.classifier = nn.Sequential(
            nn.Conv2d(ff2 * (depth - 1), num_filters, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters, 1, kernel_size=1, padding="same"),
            nn.Sigmoid(),
        )

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        results = []
        for i, layer in enumerate(self.downsampling):
            x = layer(x)
            results.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = self.bottleneck(x)

        for i, layer in enumerate(self.upsampling):
            x = torch.concatenate((x, results.pop()), dim=1)
            x = layer(x)

        x = torch.concatenate((x, results.pop()), dim=1)
        return self.classifier(x)
