import os
import os.path as osp
import math
import argparse
from functools import partial
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from geodataset import GeoDataset
from guessr_model import vit_guessr
from utils import distance_loss, train_epoch, eval_epoch, plot_loss, plot_map, plot_stats


def main(args):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps:0'
    device = torch.device(device)

    print(f'Training with {device}\n')
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = GeoDataset(args.input_dir, 'train', train_transform)
    valid_dataset = GeoDataset(args.input_dir, 'test', test_transform)
    train_size, valid_size = len(train_dataset), len(valid_dataset)

    BATCH_SIZE = args.batch_size
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print(f'train_size: {train_size} and valid_size: {valid_size}\n')

    guessr = vit_guessr(depth=args.depth, num_heads=args.num_heads).to(device)

    numparams = sum(param.numel() for param in guessr.parameters())
    print(f'Total trainable parameters: {numparams}\n')

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(guessr.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(guessr.parameters(), lr=args.lr, momentum=0.9)

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
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_model', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
