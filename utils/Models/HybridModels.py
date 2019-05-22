
import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
import torch
import torch.nn.functional as F


class Hybrid_5Conv3FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(Hybrid_5Conv3FC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding =2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384,  kernel_size=3, stride=1, padding =1),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = FlattenLayer(6 * 6 * 256)
        self.fc1 = BBBLinearFactorial(6* 6 * 256, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.flatten, self.fc1, self.dropout1, self.fc2, self.dropout2, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        x = self.features(x)
        x = x.view(x.size(0), -1)
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl


class Hybrid_5Conv1FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(Hybrid_5Conv1FC, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = BBBLinearFactorial(256, outputs)

        layers = [ self.fc1]
        self.layers = nn.ModuleList(layers)

    def probforward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl


class Hybrid_timeseries_1Conv2FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(Hybrid_timeseries_1Conv2FC, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),

        )

        self.flatten = FlattenLayer(1 * 1 * 64)
        self.fc1 = BBBLinearFactorial(1* 1 * 64, 512)
        self.fc2 = BBBLinearFactorial(512, outputs)

        layers = [self.flatten, self.fc1, self.fc2 ]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl


class Hybrid_6Conv3FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(Hybrid_6Conv3FC, self).__init__()

        self.features = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=inputs, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = FlattenLayer(4 * 4 * 256)
        self.fc1 = BBBLinearFactorial(4 * 4 * 256, 1024)
        self.soft1 = nn.Softplus()
        self.fc2 = BBBLinearFactorial(1024, 512)
        self.soft2 = nn.Softplus()
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = BBBLinearFactorial(512, outputs)


        layers = [ self.flatten,self.fc1,self.soft1,self.fc2,self.soft2,self.dropout2,self.fc3]
        self.layers = nn.ModuleList(layers)

    def probforward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl