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
from guessr_model import vit_guessr
from utils import distance_loss, train_epoch, eval_epoch, plot_loss, plot_map, plot_stats


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

    guessr = vit_guessr(depth=args.depth, num_heads=args.num_heads).to(device)
    optimizer = optim.Adam(guessr.parameters(), lr=args.lr)
    criterion = partial(distance_loss, R=1)

    train_loss, valid_loss = [], []

    for epoch in tqdm(range(args.epochs)):
        train_running_loss = train_epoch(guessr, optimizer, train_loader, criterion, device, epoch)
        valid_running_loss = eval_epoch(guessr, valid_loader, criterion, device, epoch)
        train_loss.append(train_running_loss)
        valid_loss.append(valid_running_loss)
    
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.save_model:
        torch.save(guessr.state_dict(), osp.join(args.out_dir, f'ViTGuessr_depth{args.depth}_heads_{args.num_heads}_epochs_{args.epochs}.pth'))

    plot_loss(args.out_dir, train_loss, valid_loss, args.depth, args.num_heads, args.epochs)

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

    plot_map(args.out_dir, ground_truth.copy(), prediction.copy(), args.depth, args.num_heads, args.epochs)
    plot_stats(args.out_dir, ground_truth.copy(), prediction.copy(), args.depth, args.num_heads, args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--label_name', type=str, default='coords_date.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=977)
    parser.add_argument('--raw_data', action='store_true', default=False)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--save_model', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
