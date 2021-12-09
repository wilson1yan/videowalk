import torch
import torch.nn as nn
import torch.nn.functional as F


def small_resnet():
    return SmallResNet()


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode='reflect'))


class ResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(n_hiddens),
            nn.ReLU(),
            SamePadConv2d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm2d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv2d(n_hiddens // 2, n_hiddens, 1, bias=False)
        )

    def forward(self, x):
        return x + self.model(x)

        
class SmallResNet(nn.Module):
    def __init__(self, base_channels=32, stage_sizes=[2, 2, 2, 2]):
        super().__init__()
        self.stage_sizes = stage_sizes
        self.base_channels = base_channels

        prev_channels = 3
        self.blocks = nn.ModuleDict()
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                channels = base_channels * 2 ** i
                if j == 0:
                    stride = 2 if i > 0 else 1
                    block = []
                    if i > 0:
                        block.extend([
                            nn.BatchNorm2d(prev_channels),
                            nn.ReLU()
                        ])
                    block = [SamePadConv2d(prev_channels, channels, 3, stride),
                             ResidualBlock(channels)]
                    self.blocks[f'{i}_{j}'] = nn.Sequential(*block)
                else:
                    self.blocks[f'{i}_{j}'] = ResidualBlock(channels)
            prev_channels = channels

    def forward(self, x):
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                x = self.blocks[f'{i}_{j}'](x)
        return x
