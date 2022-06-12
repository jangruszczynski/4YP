# UNet preparation
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
import albumentations as A
import torchvision.models as models
import torch.nn as nn
# progress bar
from tqdm import tqdm
import torch.optim as optim
import segmentation_models_pytorch as smp
from shutil import copyfile


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

"""
ENCODER = 'efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['gaze_points']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
"""

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
select_classes = ['points']
DEVICE = 'cuda'

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(select_classes),
    activation=ACTIVATION,
)


"""
# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
"""

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

images_dir = r'data\Cropped_Frames'
masks_dir = r'data\Masks_Gauss_no10_sigma20'

images_valid = r'data\valid\frames'
masks_valid = r'data\valid\masks'
images_train = r'data\train\frames'
masks_train = r'data\train\masks'



def divide_train_valid(images_dir, masks_dir, images_valid, masks_valid, images_train, masks_train):

    images = os.listdir(masks_dir)
    idx_ = int(0.8 * len(images))
    train = images[0:idx_]
    valid = images[idx_:]


    if len(os.listdir(images_valid)) == 0 and len(os.listdir(images_train)) == 0:
        for fn in train:
            copyfile(os.path.join(images_dir, fn), os.path.join(images_train, fn))
            copyfile(os.path.join(masks_dir, fn), os.path.join(masks_train, fn))

        for fn in valid:
            copyfile(os.path.join(images_dir, fn), os.path.join(images_valid, fn))
            copyfile(os.path.join(masks_dir, fn), os.path.join(masks_valid, fn))
    else:
        pass

divide_train_valid(images_dir, masks_dir, images_valid, masks_valid, images_train, masks_train)


train_dataset = Dataset(
    images_train,
    masks_train,
    #augmentation=get_training_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    images_valid,
    masks_valid,
    #augmentation=get_validation_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.001),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs

max_score = 0

losses = []

#writer = SummaryWriter()

# train model for 10 epochs

for i in range(0, 10):

    print('\nEpoch: {}'.format(i))
    torch.cuda.empty_cache()
    train_logs = train_epoch.run(train_loader)
    torch.cuda.empty_cache()
    valid_logs = valid_epoch.run(valid_loader)

    losses.append(loss.item())

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './UNet_model_13042021.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')



def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, sigma=5)
    plt.show()