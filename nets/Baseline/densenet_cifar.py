import torch.nn as nn
import math
from nets.modules import Cat


class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes, track_running_stats=False, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.cat = Cat()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.cat(out, x)
        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, track_running_stats=False, affine=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, depth=22, block=Bottleneck,
                 num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 6

        self.growthRate = growthRate

        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)

        self.bn = nn.BatchNorm2d(self.inplanes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(self.inplanes, num_classes)

    def _make_denseblock(self, block, blocks):
        layers = []
        for _ in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class DenseNet40(DenseNet):
    def __init__(self, num_classes=10):
        super(DenseNet40, self).__init__(depth=40, block=Bottleneck, num_classes=num_classes,
                                            growthRate=12, compressionRate=2)
