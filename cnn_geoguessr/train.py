import os
import os.path as osp
import argparse
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from geodataset import GeoDataset, rawGeoDataset
from guessr_model import cnn_guessr_baseline, cnn_guessr_panorama_padding
from utils import distance_loss, train_epoch, eval_epoch, plot_loss, plot_map, plot_stats


def main(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        try:
            if torch.backends.mps.is_available():
                device = 'mps:0'
        except:
            device = 'cpu'
    device = torch.device(device)

    print(f'Training with {device}')
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if args.raw_data:
        geo_dataset = rawGeoDataset(args.root_dir, args.img_w, args.img_h, args.label_name, img_transform)
    else:
        geo_dataset = GeoDataset(args.root_dir, args.img_w, args.img_h, args.label_name, img_transform)
    dataset_size = len(geo_dataset)
    train_size = int(dataset_size * args.train_ratio)
    valid_size = dataset_size - train_size

    BATCH_SIZE = args.batch_size
    torch.manual_seed(args.seed)

    train_geo_dataset, valid_geo_dataset = random_split(geo_dataset, [train_size, valid_size])
    # num_workers = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    num_workers = 0
    train_loader = DataLoader(train_geo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_geo_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print(f'dataset_size: {dataset_size}, randomly split into train_size: {train_size} and valid_size: {valid_size}')

    if args.panorama_padding:
        pad = 'panorama_padding'
        guessr = cnn_guessr_panorama_padding().to(device)
    else:
        pad = 'baseline'
        guessr = cnn_guessr_baseline().to(device)
    # optimizer = optim.Adam(guessr.parameters(), lr=args.lr)
    optimizer = optim.SGD(guessr.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    criterion = partial(distance_loss, R=args.radius)

    train_loss, valid_loss = [], []

    for epoch in tqdm(range(args.epochs)):
        train_running_loss = train_epoch(guessr, optimizer, train_loader, criterion, device, epoch)
        valid_running_loss = eval_epoch(guessr, valid_loader, criterion, device, epoch)
        train_loss.append(train_running_loss)
        valid_loss.append(valid_running_loss)

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.save_model:
        torch.save(guessr.state_dict(), osp.join(args.out_dir, f'CNNGuessr_{pad}_epochs_{args.epochs}.pth'))

    plot_loss(args.out_dir, train_loss, valid_loss, pad, args.epochs)

    guessr.eval()
    ground_truth, prediction = [], []

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            image, coord = data
            image, coord = Variable(image).to(device), Variable(coord).to(device)
            output = guessr(image)
            ground_truth.append(coord.cpu().numpy())
            prediction.append(output.cpu().numpy())
    
    ground_truth, prediction = np.concatenate(ground_truth), np.concatenate(prediction)

    plot_map(args.out_dir, ground_truth.copy(), prediction.copy(), pad, args.epochs)
    plot_stats(args.out_dir, ground_truth.copy(), prediction.copy(), pad, args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--panorama_padding', action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--label_name', type=str, default='coords_date.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=977)
    parser.add_argument('--raw_data', action='store_true', default=False)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--save_model', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
