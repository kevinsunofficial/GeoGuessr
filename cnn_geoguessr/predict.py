import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable

from guessr_model import cnn_guessr
from utils import geo_distance, resize_img


def main(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps:0'
    device = torch.device(device)

    print(f'Training with {device}')

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_path = args.img_path
    assert osp.exists(img_path), f'File {img_path} does not exist.'

    img = resize_img(img_path, args.img_w, args.img_h)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model_path = args.model_path
    assert osp.exists(model_path), f'File {model_path} does not exist.'

    guessr = cnn_guessr().to(device)
    guessr.load_state_dict(torch.load(model_path, map_location=device))
    guessr.eval()

    with torch.no_grad():
        output = torch.squeeze(guessr(Variable(img).to(device))).cpu().numpy()
    
    pred_coord = output * np.array([90, 180])
    ns = 'N' if pred_coord[0] > 0 else 'S'
    we = 'E' if pred_coord[1] > 0 else 'W'
    
    print(f'Predicted location: {abs(pred_coord[0]):.3}{ns}, {abs(pred_coord[1]):.3}{we}')

    if args.actual_lat is not None and args.actual_lng is not None:
        actual_coord = np.array([args.actual_lat, args.actual_lng])
        distance = geo_distance(pred_coord, actual_coord, std=False)
        print(f'Distance from actual location: {distance:.3}km')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--actual_lat', type=float, default=None)
    parser.add_argument('--actual_lng', type=float, default=None)

    args = parser.parse_args()

    main(args)
