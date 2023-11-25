import torch
import torch.nn as nn


class CNNGuessr(nn.Module):
    def __init__(self, conv, conv_out, hidden_classes, drop_ratio=0.5, 
                 img_w=256, img_h=128, in_c=3, padding=1, init_weight=False):
        super(CNNGuessr, self).__init__()

        self.img_size = (img_h, img_w)
        self.conv = conv
        self.fc = nn.Sequential(
            nn.Linear(conv_out, hidden_classes),
            nn.ReLU(True),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_classes, hidden_classes),
            nn.ReLU(True),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_classes, 2)
        )

        if init_weight:
            self.apply(_init_cnn_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
            f'Input image size ({H}, {W}) mismatch with model requirements {self.img_size}'
        
        x = self.conv(x)
        x = x.view(-1, self.conv_out)
        x = torch.tanh(self.fc(x))

        return x
    

LAYER_STRUCTURE = {
    'baseline': [32, 'M', 64, 'M', 64, 'M', 32, 'M', 16, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

CONV_OUT_DIM = {
    'baseline': 16 * 4 * 8,
    'vgg16': 512 * 4 * 8,
}

HIDDEN_DIM = {
    'baseline': 100,
    'vgg16': 4096
}


def _init_cnn_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def make_layers(layer_params, in_c=3):
    layers = []
    in_c = in_c
    for param in layer_params:
        if param == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.append(nn.Conv2d(in_c, param, 3, padding=1))
            layers.append(nn.ReLU(True))
            in_c = param
    
    return nn.Sequential(*layers)


def cnn_guessr(model):
    assert model in LAYER_STRUCTURE, f'Model {model} not in preset'
    
    guessr = CNNGuessr(conv=make_layers(LAYER_STRUCTURE[model]), 
                       conv_out=CONV_OUT_DIM[model], 
                       hidden_classes=HIDDEN_DIM[model])

    return guessr
