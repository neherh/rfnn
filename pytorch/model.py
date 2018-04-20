import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 32, (5, 5), (1, 1), (2, 2), bias = False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1),bias = False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1),bias = False)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1),bias = False)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, (3, 3), (1, 1), (1, 1),bias = False)
        self.conv5_bn = nn.BatchNorm2d(2)
        # self.softmax = nn.Softmax2d()
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        # x = x.type(torch.FloatTensor)
        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.relu(self.conv4_bn(self.conv4(x)))
        x = self.sig(self.conv5(x))
        # x = self.softmax(x)
        # x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv5.weight, init.calculate_gain('sigmoid'))
        #init.orthogonal(self.conv4.weight)

