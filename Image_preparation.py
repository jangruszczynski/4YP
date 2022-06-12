# Data input and preparation for NN
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


# Reading gaze data

header_raw = pd.read_csv('data/gaze_synchrone.csv', sep=' ', header=0, skiprows=2, nrows=0)
gaze_data = pd.read_csv('data/gaze_synchrone.csv', sep=' ', header=0, skiprows=3)

header_raw = header_raw.columns[1:33]

gaze_data.columns = header_raw

# Sorting columns needed for mapping points
gaze_coordinates = gaze_data[['right_eye.gaze_point.position_on_display_area.x()', 'right_eye.gaze_point.position_on_display_area.y()', 'left_eye.gaze_point.position_on_display_area.x()', 'left_eye.gaze_point.position_on_display_area.y()']]

gaze_coordinates.columns = ['right_x', 'right_y', 'left_x', 'left_y']

# Replacing NaNs with empty values
gaze_coordinates = gaze_coordinates.replace('-nan(ind)', '')
gaze_coordinates = gaze_coordinates.replace(r'^\s*$', 0, regex=True)
gaze_coordinates = gaze_coordinates.to_numpy().astype(float)

# Get image size

temp_image = PIL.Image.open(r'data\Frames\frame000501.png')
(img_width, img_height) = temp_image.size
print ('image size:',(img_width, img_height))
print(type(img_width))
gaze_coordinates[:,[1,3]] *= img_height
gaze_coordinates[:,[0,2]] *= img_width

# Read images
for img_no in range(5030, 5031):
    img = mpimg.imread(r'data\Frames\frame000505.png')

    x = [gaze_coordinates[img_no][0], gaze_coordinates[img_no][2]]
    y = [gaze_coordinates[img_no][1], gaze_coordinates[img_no][3]]

    plt.scatter(x, y)

plt.imshow(img)
plt.show()

# adjusting coordinates to cropped image

left_x = 357
right_x = 428
bottom_y = 213

# one coordinate (x1)

for i in range(gaze_coordinates.shape[0]):

    if gaze_coordinates[i, 0] > img_width - right_x:
        gaze_coordinates[i, 0] = np.nan
    elif gaze_coordinates[i, 0] < left_x:
        gaze_coordinates[i, 0] = np.nan
    else:
        gaze_coordinates[i, 0] -= left_x

# x2 coordinate

for i in range(gaze_coordinates.shape[0]):

    if gaze_coordinates[i, 2] > img_width - right_x:
        gaze_coordinates[i, 2] = np.nan
    elif gaze_coordinates[i, 2] < left_x:
        gaze_coordinates[i, 2] = np.nan
    else:
        gaze_coordinates[i, 2] -= left_x

# y1 coordinate

for i in range(gaze_coordinates.shape[0]):
    if gaze_coordinates[i, 1] < bottom_y:
        gaze_coordinates[i, 1] = np.nan
    else:
        gaze_coordinates[i, 1] -= bottom_y

# y2 coordinate

for i in range(gaze_coordinates.shape[0]):
    if gaze_coordinates[i, 3] < bottom_y:
        gaze_coordinates[i, 3] = np.nan
    else:
        gaze_coordinates[i, 3] -= bottom_y



# saving changed gaze data to csv file

#pd.DataFrame(gaze_coordinates).to_csv("data/gaze_data_adjusted.csv")

gaze_coordinates = np.nan_to_num(gaze_coordinates)
"""
means = []
for i in range(gaze_coordinates.shape[0]):
    means.append(np.mean(gaze_coordinates[i]))

plt.hist(means, bins=100, label='Average coordinates')
plt.legend()
plt.show()

"""

img_names = os.listdir(r'C:\Users\jgrus\PycharmProjects\Image preparation\data\Frames')[1:]

data = pd.DataFrame({'right_x': gaze_coordinates[0:len(img_names),0],
                    'right_y': gaze_coordinates[0:len(img_names),1],
                    'left_x': gaze_coordinates[0:len(img_names),2],
                    'left_y': gaze_coordinates[0:len(img_names),3],
                    'image_name': img_names}
                   )



list_files = os.listdir(r'C:\Users\jgrus\PycharmProjects\Image preparation\data\Cropped_Frames_useless')[1:]
list_files = pd.DataFrame(list_files)
list_files.columns = ['image_name']

data = data[~data.image_name.isin(list_files.image_name)]

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def create_mask(data_df: pd.DataFrame, data_dir: str, mask_dir: str, img_name: str) -> None:
    """
    data_df : Data frame for a batch of images
    data_dir : directory with images
    mask_dir : direcory where the masks are saved
    img_name : name of the base image for the mask
    """

    data = data_df.copy()
    img_path = os.path.join(data_dir, img_name)
    image = Image.open(img_path)
    width, height = image.size
    data[["right_x", "left_x"]] *= width
    data[["right_y", "left_y"]] *= height
    data[['right_x', 'right_y', 'left_x', 'left_y']] = data[['right_x', 'right_y', 'left_x', 'left_y']].astype(int)

    right = data.loc[:, ["right_x", "right_y"]]
    left = data.loc[:, ["left_x", "left_y"]]

    right = tuple(tuple(x) for x in right.values)
    left = tuple(tuple(x) for x in left.values)

    img = Image.new("1", (width, height))
    draw = ImageDraw.Draw(img)
    draw.polygon(right, fill=(1))
    draw.polygon(left, fill=(1))
    mask_path = os.path.join(mask_dir, img_name)
    img.save(mask_path)
    del img
    del draw


def create_mask_gauss(data_df: pd.DataFrame, data_dir: str, mask_dir: str, img_name: str) -> None:
    """
    data_df : Data frame for a batch of images
    data_dir : directory with images
    mask_dir : direcory where the masks are saved
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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['background', 'gaze_points']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

from shutil import copyfile

images_dir = r'data\Frames'
masks_dir = r'data\MasksGauss'

images_valid = r'data\valid\frames'
masks_valid = r'data\valid\masks'
images_train = r'data\train\frames'
masks_train = r'data\train\masks'
