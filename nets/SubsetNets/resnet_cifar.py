import torch.nn as nn

from nets.Baseline.resnet_cifar import ResNet as BaseResNet
from nets.Baseline.resnet_cifar import BasicBlock as BaseBasicBlock
from nets.modules import Add


class BasicBlock(BaseBasicBlock):
    def __init__(self, in_planes, planes, stride=1, subset_factor=1.0):
        super().__init__(in_planes, planes, stride)
        self.add = Add()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        out = self.relu2(out)
        return out


class ResNet(BaseResNet):
    def __init__(self, block, num_blocks, num_classes=10, subset_factor=1.0):
        super(BaseResNet, self).__init__()
        self._register_load_state_dict_pre_hook(self.sd_hook)

        self.subset_factor = subset_factor
        self.in_planes = int(16*subset_factor)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, track_running_stats=False, momentum=None)
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(block, int(16*subset_factor), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32*subset_factor), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64*subset_factor), num_blocks[2], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(int(256*subset_factor), num_classes)

        self.flatten = nn.Flatten()

    def sd_hook(self, state_dict, *_):
        pass

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, subset_factor=self.subset_factor))
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
    def __init__(self, subset_factor=1.0, num_classes=10):
        super().__init__(BasicBlock, [3, 3, 3], num_classes=num_classes, subset_factor=subset_factor)


class ResNet44(ResNet):
    def __init__(self, subset_factor=1.0, num_classes=10):
        super().__init__(BasicBlock, [7, 7, 7], num_classes=num_classes, subset_factor=subset_factor)
        self.linear = nn.Linear(int(4*256*subset_factor), num_classes)


class ResNet56(ResNet):
    def __init__(self, subset_factor=1.0, num_classes=10):
        super().__init__(BasicBlock, [9, 9, 9], num_classes=num_classes, subset_factor=subset_factor)
        self.linear = nn.Linear(int(4*256*subset_factor), num_classes)
