import torch.nn as nn
import torch


class Cat(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class SLTShortcut(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, target_tensor):
        return x[:, 0:target_tensor.shape[1], :, :]
