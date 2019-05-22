import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.BBBlayers import FlattenLayer

def conv_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        #nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        #nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)


class AlexNet_5Conv1FC(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet_5Conv1FC, self).__init__()
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
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNetTimeSeries_1Conv2FC(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNetTimeSeries_1Conv2FC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            #nn.Conv1d(64, 192, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
        )
        #self.classifier = nn.Linear(192, num_classes)
        self.classifier = nn.Sequential(

            nn.Linear(64, 512),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet_5Conv3FC(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet_5Conv3FC, self).__init__()
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
        self.classifier = nn.Sequential(
            FlattenLayer(6 * 6 * 256),
            nn.Linear(6* 6 * 256, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet_5Conv3FCwithDropout(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet_5Conv3FCwithDropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding =2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384,  kernel_size=3, stride=1, padding =1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(6 * 6 * 256),
            nn.Linear(6* 6 * 256, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNet_6Conv3FC(nn.Module):
    """AlexNetCifar10."""

    def __init__(self, num_classes, inputs=3):
        """AlexNetCifar10 Builder."""
        super(AlexNet_6Conv3FC, self).__init__()

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

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.features(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.classifier(x)

        return x


class AlexNet_3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(AlexNet_3Conv3FC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            FlattenLayer(2 * 2 * 128),
            nn.Linear(2 * 2 * 128, 1000),
            nn.Softplus(),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print('X', x)
        return x