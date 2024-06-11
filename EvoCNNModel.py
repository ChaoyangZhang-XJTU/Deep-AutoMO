import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetUnit(nn.Module):
    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layer(x)
        return out

class DenseNetBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(DenseNetBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseNetUnit(nn.Module):
    def __init__(self, k, amount, in_channel, max_input_channel):
        super(DenseNetUnit, self).__init__()
        if in_channel > max_input_channel:
            self.need_conv = True
            self.bn = nn.BatchNorm2d(in_channel)
            self.conv = nn.Conv2d(in_channel, max_input_channel, kernel_size=1, bias=False)
            in_channel = max_input_channel

        self.layer = self._make_dense(in_channel, k, amount)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
    def forward(self, x):
        out = x
        if hasattr(self, 'need_conv'):
            out = self.conv(F.relu(self.bn(out)))
        out = self.layer(out)
        return out

class ModelSettings():
    min_units = 1
    max_units = 4
    min_amount = 3
    max_amount1 = 10
    max_amount2 = 5
    output_channles = [64, 128, 256]
    pool_types = [0, 1]
    k_list = [12, 20, 40]
    max_input_channels = [128, 64, 32]

class EvoCNNModel(nn.Module):
    def __init__(self, num_classes, individual):
        super(EvoCNNModel, self).__init__()
        #generated_init
        self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer, out = self._make_layer(individual)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out, num_classes)

    def _make_layer(self, individual):
        layers = []
        for i in range(len(individual)):
            if len(individual[i]) == 3:
                layers.append(ResNetUnit(amount=individual[i][0], in_channel=individual[i][1], out_channel=individual[i][2]))
                out = individual[i][2]
            if len(individual[i]) == 1:
                if individual[i][0] == 0: layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
                if individual[i][0] == 1: layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))            
            if len(individual[i]) == 4:
                layers.append(DenseNetUnit(k=individual[i][0], amount=individual[i][1], in_channel=individual[i][2], max_input_channel=individual[i][3]))
                true_input_channel = individual[i][2]
                if true_input_channel > individual[i][3]:
                    true_input_channel = individual[i][3]
                out = true_input_channel + individual[i][0] * individual[i][1]
        
        return nn.Sequential(*layers), out

    def forward(self, x):
        #generate_forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x