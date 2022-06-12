# Mask creation
# Jan Gruszczynski 4YP
# University of Oxford
# Import libraries

import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import glob
import PIL.Image
#import cv2
import os
import numpy as np
import random
import colorsys
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


data = pd.read_csv('data/gaze_data_cropped.csv')
data = data.reset_index(drop=True)[1:]
data[['right_x', 'right_y', 'left_x', 'left_y']] = data[['right_x', 'right_y', 'left_x', 'left_y']].astype(int)

#all = data.loc[:, ['right_x', 'right_y', 'left_x', 'left_y']]

#right = data.loc[:, ["right_x", "right_y"]]
#left = data.loc[:, ["left_x", "left_y"]]
#right = tuple(tuple(x) for x in right.values)
#left = tuple(tuple(x) for x in left.values)


#impulses = np.zeros((600,800))

#all = tuple(tuple(x) for x in np.reshape(all.values, (-1, 2)))
#all = np.asarray(all)


def create_mask_polygon(data_df: pd.DataFrame, data_dir: str, mask_dir: str, img_name: str) -> None:
    """
    data_df : Data frame for a batch of images
    data_dir : directory with images
    mask_dir : directory where the masks are saved
    img_name : name of the base image for the mask
    """
    data = data_df.copy()
    img_path = os.path.join(data_dir, img_name)
    image = Image.open(img_path)
    width, height = image.size
    data[['right_x', 'right_y', 'left_x', 'left_y']] = data[['right_x', 'right_y', 'left_x', 'left_y']].astype(int)
    right = data.loc[:, ["right_x", "right_y"]]
    left = data.loc[:, ["left_x", "left_y"]]
    right = tuple(tuple(x) for x in right.values)
    left = tuple(tuple(x) for x in left.values)
    all = data.loc[:, ['right_x', 'right_y', 'left_x', 'left_y']]
    all = tuple(tuple(x) for x in np.reshape(all.values, (-1, 2)))

    img = Image.new("1", (width, height))
    draw = ImageDraw.Draw(img)
    draw.polygon(all, fill=(1))
    mask_path = os.path.join(mask_dir, img_name)
    img.save(mask_path)
    del img
    del draw

data_dir = r'data\Cropped_Frames'
mask_dir = r'data\Masks_Gauss_1'
"""
for i in tqdm(range(data.shape[0]//10),position=0, leave=True):
    data_ = data.loc[10*i:10*(i+1) - 1,:]
    img_name = data.loc[(10*i+10*(i+1))//2, "image_name"]
    result = create_mask_polygon(data_.fillna(0),data_dir, mask_dir, img_name)
"""




def create_mask_gauss(data_df: pd.DataFrame, data_dir: str, mask_dir: str, img_name: str) -> None:
    """
    data_df : Data frame for a batch of images
    data_dir : directory with images
    mask_dir : directory where the masks are saved
    img_name : name of the base image for the mask
    """
    sigma = 25  # spread of Gaussian
    img_path = os.path.join(data_dir, img_name)
    image = Image.open(img_path)
    width, height = image.size
    data_df[["right_x", "left_x"]] *= width
    data_df[["right_y", "left_y"]] *= height
    data_df[['right_x', 'right_y', 'left_x', 'left_y']] = data_df[['right_x', 'right_y', 'left_x', 'left_y']].astype(
        int)
    impulses = np.zeros(image.size)
    right = data_df.loc[:, ["right_x", "right_y"]]
    left = data_df.loc[:, ["left_x", "left_y"]]
    right = np.array([list(x) for x in right.values])
    left = np.array([list(x) for x in left.values])
    gazes = np.vstack([right, left])
    impulses[np.clip(gazes[:, 0], 0, width - 1), np.clip(gazes[:, 1], 0, height - 1)] = 1
    result = gaussian_filter(impulses.T, sigma, mode='nearest')
    result = (result > 0).astype(np.uint8)
    mask_path = os.path.join(mask_dir, img_name)
    cv2.imwrite(mask_path, result)

def create_mask_gauss_distribution(data_df: pd.DataFrame, data_dir: str, mask_dir: str, img_name: str, sigma: int) -> None:
    """
    data_df : Data frame for a batch of images
    data_dir : directory with images
    mask_dir : directory where the masks are saved
    img_name : name of the base image for the mask
    sigma : spread of gaussian
    """
    img_path = os.path.join(data_dir, img_name)
    image = Image.open(img_path)

    width, height = image.size
    data_df[['right_x', 'right_y', 'left_x', 'left_y']] = data_df[['right_x', 'right_y', 'left_x', 'left_y']].astype(
        int)
    image_h, image_w = image.size

    new_size = image_w, image_h

    data_df[['right_y', 'left_y']] = image_w - data_df[['right_y', 'left_y']] - 1

    # array same size of image
    impulses = np.zeros(new_size)

    # rows and cols are the row and column indices of the centers
    # of the gaussian peaks.
    # np.random.seed(123456)
    # this chooses random points but you will define from gaze point coordinates
    # rows, cols = np.unravel_index(np.random.choice(impulses.size, replace=False, size=num_centers), impulses.shape)
    # set pixel coordinate values to 1
    # impulses[(500, 600), (500, 600)] = 1
    # or use this if you want duplicates to sum:
    # np.add.at(impulses, (rows, cols), 1)

    impulses_right_1 = np.asarray(data_df[['right_x', 'right_y']])[0:3, 1]
    impulses_right_2 = np.asarray(data_df[['right_x', 'right_y']])[0:3, 0]

    impulses_left_1 = np.asarray(data_df[['left_x', 'left_y']])[0:3, 1]
    impulses_left_2 = np.asarray(data_df[['left_x', 'left_y']])[0:3, 0]

    impulses[impulses_right_1, impulses_right_2] = 1

    impulses[impulses_left_1,impulses_left_2] = 1

    # filter impulses to create the result.
    result = gaussian_filter(impulses, sigma, mode='nearest')

    mask_path = os.path.join(mask_dir, img_name)
    #cv2.imwrite(mask_path, result)

    rescaled = (255.0 / result.max() * (result - result.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(mask_path)


data_dir = r'data\Cropped_Frames'
mask_dir = r'data\Masks_Gauss_no10_sigma20'

#img_name = data.loc[1, "image_name"]
#result = create_mask_gauss_distribution(data.fillna(0),data_dir, mask_dir, img_name, 25)

no_frames = 10

for i in tqdm(range(data.shape[0]//no_frames),position=0, leave=True):
    data_ = data.loc[no_frames*i:no_frames*(i+1) - 1,:]
    data_ = data_.reset_index(drop=True)
    img_name = data.loc[(no_frames*i+no_frames*(i+1))//2, "image_name"]
    result = create_mask_gauss_distribution(data_.fillna(0),data_dir, mask_dir, img_name, 20)



def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
