import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


def distance_loss(output, target, R=6371):
    rad_output, rad_target = torch.deg2rad(output), torch.deg2rad(target)
    lat1, lng1 = rad_output[:, 0], rad_output[:, 1]
    lat2, lng2 = rad_target[:, 0], rad_target[:, 1]
    dlat, dlng = lat1 - lat2, lng1 - lng2
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    c = 2 * torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))
    loss = R * c

    return torch.mean(loss)


def geo_distance(points1, points2, R=6371, std=False):
    if std:
        mult = np.array([90, 180])
        points1, points2 = points1 * mult, points2 * mult
    rad1, rad2 = np.radians(points1), np.radians(points2)
    lat1, lng1 = rad1[:, 0], rad1[:, 1]
    lat2, lng2 = rad2[:, 0], rad2[:, 1]
    dlat, dlng = lat1 - lat2, lng1 - lng2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c

    return dist


def train_epoch(model, optimizer, data_loader, criterion, device, epoch):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.
    num_data = 0
    
    for i, data in enumerate(data_loader, 0):
        image, coord = data
        image, coord = Variable(image).to(device), Variable(coord).to(device)
        output = model(image)
        loss = criterion(output, coord)
        loss.backward()

        running_loss += loss.detach().cpu().numpy()
        num_data += 1
        optimizer.step()
        optimizer.zero_grad()
    
    return running_loss / num_data


@torch.no_grad()
def eval_epoch(model, data_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.
    num_data = 0
    
    for i, data in enumerate(data_loader, 0):
        image, coord = data
        image, coord = Variable(image).to(device), Variable(coord).to(device)
        output = model(image)
        loss = criterion(output, coord)

        running_loss += loss.detach().cpu().numpy()
        num_data += 1
    
    return running_loss / num_data


def plot_loss(plot_dir, train_loss, valid_loss, pad, epochs):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='training loss')
    plt.plot(valid_loss, label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Standardized distance loss')
    plt.title(f'Training stats with {epochs} epochs')

    plt.savefig(osp.join(plot_dir, f'loss_{pad}_epochs_{epochs}.png'))
    plt.clf()


def plot_map(plot_dir, ground_truth, prediction, pad, epochs):
    mult = np.array([90, 180])
    ground_truth_real, prediction_real = ground_truth.copy() * mult, prediction.copy() * mult

    plt.figure(figsize=(8, 5))
    plt.scatter(ground_truth_real[:, 1], ground_truth_real[:, 0], 
                label='ground truth', alpha=0.1, color='blue', s=5)
    plt.scatter(prediction_real[:, 1], prediction_real[:, 0], 
                label='prediction', alpha=0.1, color='red', s=5)
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Validation map with {epochs} epochs')

    plt.savefig(osp.join(plot_dir, f'valid_map_{pad}_epochs_{epochs}.png'))
    plt.clf()


def plot_stats(plot_dir, ground_truth, prediction, pad, epochs):
    distances = geo_distance(ground_truth.copy(), prediction.copy(), std=True)
    n = distances.size

    grid = ['Street', 'City', 'Region', 'Country', 'Continent']
    dist = [1, 25, 250, 750, 2500]
    proportion = []
    
    for d in dist:
        proportion.append((distances <= d).sum() / n)

    df = pd.DataFrame({
        'Range': grid, 'Distances': dist, 'Proportion': proportion
    })

    print(df)
    df.to_csv(osp.join(plot_dir, f'distr_dist_{pad}_epochs_{epochs}.csv'), header=True, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=250)
    plt.xlabel('Distances (km)')
    plt.ylabel('Count')
    plt.title(f'Error distances (km) with {epochs} epochs')

    plt.savefig(osp.join(plot_dir, f'distr_dist_{pad}_epochs_{epochs}.png'))
    plt.clf()
