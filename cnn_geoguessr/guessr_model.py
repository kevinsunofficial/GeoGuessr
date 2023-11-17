import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import partial


class CNNGuessr(nn.Module):
    def __init__(self, img_w=256, img_h=128, in_c=3, padding=0, panorama_padding=0, panorama_padder=None, 
                 conv_out=16*2*6, hidden_classes=100, num_out=2, drop_ratio=0.3):
        super().__init__()

        self.img_size = (img_h, img_w)
        self.padding = padding
        self.panorama_padding = panorama_padding
        self.panorama_padder = panorama_padder
        self.conv_out = conv_out
        if self.panorama_padding:
            assert self.panorama_padder is not None, \
                f'panorama_padding: {self.panorama_padding}, but have no panorama padder.'
            self.panorama_padder = partial(panorama_padder, pad_size=self.panorama_padding)
        else:
            self.panorama_padder = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=self.padding),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        self.drop = nn.Dropout(drop_ratio)
        self.fc1 = nn.Linear(self.conv_out, hidden_classes)
        self.fc2 = nn.Linear(hidden_classes, num_out)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
            f'Input image size ({H}, {W}) mismatch with model requirements {self.img_size}'
        
        x = self.conv1(self.panorama_padder(x))
        x = self.conv2(self.panorama_padder(x))
        x = self.conv3(self.panorama_padder(x))
        x = self.conv4(self.panorama_padder(x))
        x = self.conv5(self.panorama_padder(x))

        x = x.view(-1, self.conv_out)
        x = F.relu(self.fc1(self.drop(x)))
        x = torch.tanh(self.fc2(self.drop(x)))

        return x


def panorama_padder(x, pad_size):
    left_pad = x[:, :, :, -pad_size:]
    right_pad = x[:, :, :, :pad_size]
    x = torch.cat((left_pad, x, right_pad), dim=3)

    return x


def cnn_guessr_baseline():
    guessr = CNNGuessr(img_w=256, img_h=128, in_c=3, padding=1, 
                       panorama_padding=0, panorama_padder=None,
                       conv_out=16*4*8, hidden_classes=100, num_out=2)
    
    return guessr


def cnn_guessr_panorama_padding():
    guessr = CNNGuessr(img_w=256, img_h=128, in_c=3, padding=1, 
                       panorama_padding=1, panorama_padder=panorama_padder,
                       conv_out=16*4*9, hidden_classes=100, num_out=2)
    
    return guessr
