
import torch.nn as nn
from utils.BBBlayers import BBBConv2d,BBBConv1d, BBBLinearFactorial, FlattenLayer



class BBBAlexNet_5Conv1FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNet_5Conv1FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 64, kernel_size=11, stride=4, padding=5)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 192, kernel_size=5, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(192, 384, kernel_size=3, padding=1)
        self.soft3 = nn.Softplus()

        self.conv4 = BBBConv2d(384, 256, kernel_size=3, padding=1)
        self.soft4 = nn.Softplus()

        self.conv5 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 256)
        self.fc1 = BBBLinearFactorial(1* 1 * 256, outputs)


        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.conv4, self.soft4, self.conv5, self.soft5, self.pool3, self.flatten, self.fc1]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
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


class BBBAlexNetTimeSeries_1Conv2FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNetTimeSeries_1Conv2FC, self).__init__()

        self.conv1 = BBBConv1d(inputs, 64, kernel_size=11, stride=4, padding=5)
        self.soft1 = nn.Softplus()

        #self.conv2 = BBBConv1d(64, 192, kernel_size=5, padding=2)
        #self.soft2 = nn.Softplus()

        self.flatten = FlattenLayer(1 * 1 * 64)
        self.fc1 = BBBLinearFactorial(1* 1 * 64, 512)
        #self.fc2 = BBBLinearFactorial(4096, 4096)
        self.fc3 = BBBLinearFactorial(512, outputs)

        layers = [self.conv1, self.soft1,self.flatten, self.fc1, self.fc3 ]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
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


class BBBAlexNet_5Conv3FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNet_5Conv3FC, self).__init__()

        self.conv1 = BBBConv2d(inputs, 96, kernel_size=11, stride=4)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(96, 256, kernel_size=5, stride=1, padding =2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(256, 384, kernel_size=3, stride=1, padding =1)
        self.soft3 = nn.Softplus()

        self.conv4 = BBBConv2d(384, 384, kernel_size=3, stride=1, padding =1)
        self.soft4 = nn.Softplus()

        self.conv5 = BBBConv2d(384, 256, kernel_size=3, padding =1)
        self.soft5 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        #self.flatten = FlattenLayer(6 * 6 * 256)
        self.flatten = FlattenLayer(6 * 6 * 256)
        self.fc1 = BBBLinearFactorial(6* 6 * 256, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = BBBLinearFactorial(4096, outputs)

        #self.fc1 = BBBLinearFactorial(6* 6 * 256, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.conv4, self.soft4, self.conv5, self.soft5, self.pool3, self.flatten, self.fc1, self.fc2, self.fc3, self.dropout1,self.dropout2]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
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


class BBBAlexNet_6Conv3FC(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBAlexNet_6Conv3FC, self).__init__()


        self.conv1 = BBBConv2d(inputs, 32, kernel_size=3, stride= 1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.soft1 = nn.Softplus()
        self.conv2 = BBBConv2d(32, 64, kernel_size=3, stride = 1, padding=1)
        self.soft2 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(64, 128, kernel_size=3, stride = 1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(128, 128, kernel_size=3, stride = 1, padding=1)
        self.soft4 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.05)

        self.conv5 = BBBConv2d(128, 256, kernel_size=3, stride = 1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.soft5 = nn.Softplus()
        self.conv6 = BBBConv2d(256, 256, kernel_size=3, stride = 1, padding=1)
        self.soft6 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout(p=0.1)
        self.flatten = FlattenLayer(4 * 4 * 256)
        self.fc1 = BBBLinearFactorial(4 * 4 * 256, 1024)
        self.soft7 = nn.Softplus()
        self.fc2 = BBBLinearFactorial(1024, 512)
        self.soft8 = nn.Softplus()
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = BBBLinearFactorial(512, outputs)

        layers = [self.conv1,  self.soft1, self.conv2, self.soft2, self.pool1, self.conv3,
                  self.soft3, self.conv4, self.soft4, self.pool2,
                   self.conv5, self.soft5, self.conv6, self.soft6, self.pool3,
                 self.flatten, self.fc1, self.soft7, self.fc2, self.soft8, self.dropout3, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
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
