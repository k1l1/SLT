import torch.nn as nn
import math

from nets.Baseline.densenet_cifar import Bottleneck as Bottleneck_baseline
from nets.Baseline.densenet_cifar import Transition as Transition_baseline
from nets.Baseline.densenet_cifar import DenseNet as DenseNet_baseline

from nets.modules import Cat


class Bottleneck(Bottleneck_baseline):
    def __init__(self, in_planes, expansion=4, growthRate=12, subset_factor=1.0):
        super(Bottleneck_baseline, self).__init__()
        planes = int(expansion * growthRate * subset_factor)
        growthRate = int((in_planes + growthRate)*subset_factor - int(in_planes*subset_factor))
        in_planes = int(in_planes * subset_factor)

        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.cat = Cat()


class Transition(Transition_baseline):
    def __init__(self, in_planes, out_planes, subset_factor=1.0):
        super(Transition_baseline, self).__init__()
        in_planes = int(in_planes*subset_factor)
        out_planes = int(out_planes*subset_factor)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2)


class DenseNet(DenseNet_baseline):
    def __init__(self, depth=22, num_classes=10, growthRate=12, compressionRate=2, subset_factor=1.0):
        super(DenseNet_baseline, self).__init__()

        self._subset_factor = subset_factor

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 6

        self.growthRate = growthRate
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, int(self.inplanes*subset_factor), kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_denseblock(Bottleneck, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(Bottleneck, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(Bottleneck, n)

        self.bn = nn.BatchNorm2d(int(self.inplanes*subset_factor), track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(int(self.inplanes*subset_factor), num_classes)

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, growthRate=self.growthRate, subset_factor=self._subset_factor))
            self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = math.floor(self.inplanes // compressionRate)
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, subset_factor=self._subset_factor)


class DenseNet40(DenseNet):
    def __init__(self, num_classes=10, subset_factor=1.0):
        super(DenseNet40, self).__init__(depth=40, num_classes=num_classes, growthRate=12,
                                                compressionRate=2, subset_factor=subset_factor)
