import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class CNNGuessr(nn.Module):
    def __init__(self, img_w=256, img_h=128, in_c=3, hidden_classes=100, num_out=2, drop_ratio=0.1):
        super().__init__()
        self.img_size = (img_h, img_w)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        self.drop = nn.Dropout(drop_ratio)
        self.fc1 = nn.Linear(16 * 6 * 2, hidden_classes)
        self.fc2 = nn.Linear(hidden_classes, num_out)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
            f'Input image size ({H}, {W}) mismatch with model requirements {self.img_size}'
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 16 * 6 * 2)
        x = F.relu(self.fc1(self.drop(x)))
        x = F.tanh(self.fc2(self.drop(x)))

        return x


def cnn_guessr():
    guessr = CNNGuessr(img_w=256, img_h=128, in_c=3, 
                       hidden_classes=100, num_out=2, drop_ratio=0.1)

    return guessr
