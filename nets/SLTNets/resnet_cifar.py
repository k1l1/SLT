import torch.nn as nn

from nets.Baseline.resnet_cifar import ResNet as BaseResNet
from nets.modules import Add, SLTShortcut
from nets.SLTNets.hook import slt_nn_hook


class Shortcut(nn.Module):
    def __init__(self, in_planes, planes) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, track_running_stats=False, momentum=None)

    def forward(self, x, _):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, momentum=None)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, momentum=None)
        self.relu2 = nn.ReLU(inplace=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Shortcut(in_planes, planes)
        else:
            self.shortcut = SLTShortcut()

        self.add = Add()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x, out))
        out = self.relu2(out)
        return out


class ResNet(BaseResNet):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BaseResNet, self).__init__()
        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, track_running_stats=False, momentum=None)
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(256, num_classes)

        self.flatten = nn.Flatten()

    def sd_hook(self, state_dict, *_):
        slt_nn_hook(self, state_dict)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class ResNet20(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [3, 3, 3], num_classes=num_classes)


class ResNet44(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [7, 7, 7], num_classes=num_classes)
        self.linear = nn.Linear(4*256, num_classes)


class ResNet56(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [9, 9, 9], num_classes=num_classes)
        self.linear = nn.Linear(4*256, num_classes)
