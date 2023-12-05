import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image


def resize_img(img_name, img_w=256, img_h=128):
    with Image.open(img_name) as img:
        width, height = img.size
        ideal_w = height * 2
        if width > ideal_w:
            l, r = (width - ideal_w) // 2, (width + ideal_w) // 2
            img = img.crop((l, 0, r, height))
        img.thumbnail((img_w, img_h))
        array = np.array(img, dtype=np.float32)

    assert array.shape == (img_h, img_w, 3), \
        f'Image shape {array.shape} mismatch, expecting ({img_h}, {img_w}, 3)'
    
    return array


def main(args):
    labels = pd.read_csv(osp.join(args.input_dir, 'images.csv'), header=0, index_col=None)
    labels = labels.drop(11059).reset_index(drop=True) # hard-coded for single missing item
    labels.lat /= 90
    labels.lng /= 180

    img_dir = osp.join(args.input_dir, 'images')

    print(f'Split training and testing data with ratio {args.train_ratio}')
    train_labels = labels.sample(frac=args.train_ratio, random_state=args.seed)
    test_labels = labels.drop(train_labels.index)
    num_train, num_test = train_labels.shape[0], test_labels.shape[0]
    print(f'training data: {num_train}, testing data: {num_test}')

    train_images = np.array([
        resize_img(
            osp.join(img_dir, f'{train_id}.jpeg'),
            args.img_w, args.img_h
        ) for train_id in tqdm(
            train_labels.id, desc='processing training images', position=0, leave=True
        )
    ])
    test_images = np.array([
        resize_img(
            osp.join(img_dir, f'{test_id}.jpeg'),
            args.img_w, args.img_h
        ) for test_id in tqdm(
            test_labels.id, desc='processing testing images', position=0, leave=True
        )
    ])

    assert train_images.shape == (num_train, args.img_h, args.img_w, 3), \
        f'train_images shape {train_images.shape} mismatch, expecting ({num_train}, {args.img_h}, {args.img_w}, 3)'
    assert test_images.shape == (num_test, args.img_h, args.img_w, 3), \
        f'test_images shape {test_images.shape} mismatch, expecting ({num_test}, {args.img_h}, {args.img_w}, 3)'

    np.save(osp.join(args.out_dir, 'train_images.npy'), train_images)
    np.save(osp.join(args.out_dir, 'test_images.npy'), test_images)
    train_labels.to_csv(osp.join(args.out_dir, 'train_images.csv'), header=True, index=False)
    test_labels.to_csv(osp.join(args.out_dir, 'test_images.csv'), header=True, index=False)

    print(f'Preprocessing complete. Files saved to {args.out_dir}')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=977)

    args = parser.parse_args()

    main(args)
