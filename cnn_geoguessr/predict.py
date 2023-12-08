import os
import os.path as osp
import argparse
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from guessr_model import cnn_guessr
from utils import distance_loss, geo_distance, resize_img


class GeoDistanceTarget:
    def __init__(self, label, criterion):
        self.label = label
        self.criterion = criterion
    
    def __call__(self, model_output):
        return self.criterion(model_output.unsqueeze(0), self.label)


def main(args):
    device = 'cpu'
    use_cuda = False
    if torch.cuda.is_available():
        device = 'cuda'
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = 'mps:0'
    device = torch.device(device)

    print(f'Testing with {device}\n')

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_path = args.img_path
    assert osp.exists(img_path), f'File {img_path} does not exist.'

    rgb_img = resize_img(img_path)
    img = img_transform(rgb_img)
    img = torch.unsqueeze(img, dim=0).to(device)
    
    actual_coord = np.array([args.actual_lat, args.actual_lng])
    mult = np.array([90, 180])
    label = torch.tensor(actual_coord / mult, dtype=torch.float32).unsqueeze(0).to(device)

    model_path = args.model_path
    assert osp.exists(model_path), f'File {model_path} does not exist.'

    guessr = cnn_guessr('baseline').to(device)
    guessr.load_state_dict(torch.load(model_path, map_location=device))
    target_layers = [guessr.conv[-1]]
    guessr.eval()

    with torch.no_grad():
        output = torch.squeeze(guessr(Variable(img).to(device))).cpu().numpy()
    
    pred_coord = output * mult
    ns = 'N' if pred_coord[0] > 0 else 'S'
    we = 'E' if pred_coord[1] > 0 else 'W'
    
    print(f'Predicted coordinate: {pred_coord[0]:.3} ({ns}), {pred_coord[1]:.3} ({we})')

    distance = geo_distance(np.array([pred_coord]), np.array([actual_coord]), std=False)[0]
    print(f'Distance from actual location: {round(distance, 3)}km')

    cam = GradCAM(model=guessr, target_layers=target_layers, use_cuda=use_cuda)
    targets = [GeoDistanceTarget(label, partial(distance_loss, R=1))]

    greyscale_cam = cam(input_tensor=img, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img / 255., greyscale_cam, use_rgb=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    im = ax1.imshow(rgb_img / 255.)
    im = ax2.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--actual_lat', type=float, required=True)
    parser.add_argument('--actual_lng', type=float, required=True)

    args = parser.parse_args()

    main(args)
