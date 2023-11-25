import os
import os.path as osp
import math
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
import torch.optim.lr_scheduler as lr_scheduler
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
    
    if args.augment:
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(args.img_h, args.img_w), antialias=True),
            transforms.RandomPhotometricDistort(),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
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

    numparams = sum(param.numel() for param in guessr.parameters())
    print(f'Total trainable parameters: {numparams}')

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(guessr.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(guessr.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    mode = f'depth{args.depth}_heads{args.num_heads}_{args.optimizer}{args.lr}'
    criterion = partial(distance_loss, R=args.radius)

    train_loss, valid_loss = [], []

    for epoch in tqdm(range(args.epochs)):
        train_running_loss = train_epoch(guessr, optimizer, train_loader, criterion, device, epoch)
        scheduler.step()
        valid_running_loss = eval_epoch(guessr, valid_loader, criterion, device, epoch)

        train_loss.append(train_running_loss)
        valid_loss.append(valid_running_loss)
    
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    if args.save_model:
        torch.save(guessr.state_dict(), osp.join(args.out_dir, f'ViTGuessr_{mode}_epochs_{args.epochs}.pth'))

    plot_loss(args.out_dir, train_loss, valid_loss, mode, args.epochs)

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

    plot_map(args.out_dir, ground_truth.copy(), prediction.copy(), mode, args.epochs)
    plot_stats(args.out_dir, ground_truth.copy(), prediction.copy(), mode, args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--label_name', type=str, default='coords_date.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=977)
    parser.add_argument('--raw_data', action='store_true', default=False)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--save_model', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
