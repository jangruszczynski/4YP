import os
from statistics import mean

import torch
import numpy as np
import segmentation_models_pytorch as smp

import train_modified
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import matplotlib.pyplot as plt

DATA_DIR = r"data_1\Data_to_network"
ABS_DIR = r"C:\Users\jgrus\PycharmProjects\Image preparation"


x_train_dir = os.path.join(DATA_DIR, 'Frames')
y_train_dir = os.path.join(DATA_DIR, 'Masks')

x_valid_dir = os.path.join(DATA_DIR, 'Frames_valid')
y_valid_dir = os.path.join(DATA_DIR, 'Masks_valid')

x_test_dir = os.path.join(DATA_DIR, 'Frames_test')
y_test_dir = os.path.join(DATA_DIR, 'Masks_test')

# helper function for data visualization
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


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import PIL.ImageOps


class Dataset(BaseDataset):
    """PULSE Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['gaze']

    def __init__(
            self,
            abs_dir,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            validation=bool,
    ):
        if validation:
            self.ids = random.sample(os.listdir(images_dir), 100)
        else:
            self.ids = random.sample(os.listdir(images_dir), 500)
        self.images_fps = [os.path.join(abs_dir, images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(abs_dir, masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting color codes
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.bitwise_not(mask)  # inverting colors of masks

        #image_id = self.images_fps[i]

        # extract certain classes from mask
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

        return image, mask #, image_id

    def __len__(self):
        return len(self.ids)

import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=160, min_width=160, always_apply=True, border_mode=0),
        albu.RandomCrop(height=160, width=160, always_apply=True),

        #albu.IAAAdditiveGaussianNoise(p=0.2),
        #albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         #albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    
    return albu.Compose(train_transform)


"""
def get_training_augmentation():
    train_transform = [

        #albu.ToGray(p=1),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.7, border_mode=0),
        albu.PadIfNeeded(min_height=200, min_width=200, always_apply=True, border_mode=0),
        albu.RandomCrop(height=200, width=200, always_apply=True),
        albu.GaussNoise(p=0.2),

    ]

    return albu.Compose(train_transform)
"""
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(448, 448)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['gaze']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    ABS_DIR,
    x_train_dir,
    y_train_dir,
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    validation=False
)

valid_dataset = Dataset(
    ABS_DIR,
    x_valid_dir,
    y_valid_dir,
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    validation=True
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.001),
])

#print("optimizer",optimizer)

train_epoch = train_modified.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = train_modified.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 12 epochs
# with modified train module

max_score = 0

list_train_logs = []
#list_train_curve = []
list_train_metric_logs = []
list_valid_logs = []
list_valid_metric_logs = []

for i in range(0, 40):

    print('\nEpoch: {}'.format(i))
    train_logs, train_curve, metric_curve = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    ### all the logs in one place ###
    list_train_logs.append(train_logs['dice_loss'])
    #list_train_curve.append(train_curve)
    list_train_metric_logs.append(train_logs['iou_score'])
    #list_valid_logs.append(valid_logs)

    list_valid_logs.append(valid_logs[0]['dice_loss'])
    list_valid_metric_logs.append(valid_logs[0]['iou_score'])


    #list_all_train_logs.append(np.array(train_curve))
    #list_all_metric_logs.append(np.array(metric_curve))


    #list_train_logs.append(train_logs)
    #list_train_curve.append(np.mean(np.array(train_curve)))
    #list_metric_curve.append(np.mean(np.array(metric_curve)))

    #list_valid_logs.append(valid_logs)
    #train_curve_average = [l.tolist() for l in list_train_curve]
    #metric_curve_average = [l.tolist() for l in list_metric_curve]

    #list_train_curve.append(train_curve_average)
    #list_metric_curve.append(metric_curve_average)

    #print(train_curve)
    #print(type(train_curve))

    #print(list_metric_curve)
    #print(valid_logs[0]['iou_score'])

    # do something (save model, change lr, etc.)
    if max_score < valid_logs[0]['iou_score']:
        max_score = valid_logs[0]['iou_score']
        #torch.save(model, './best_model_08_05_500epoch.pth')
        print('Model saved!')

    if i == 20:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
        torch.save(model, './best_model_14_05_end_500_lr_10.pth')

    with open("full_train_curve_14.05_500_40epoch_lr_10.txt", "w") as output:
        output.write(str(list_train_logs))

    with open("full_train_metric_curve_14.05_500_40epoch_lr_10.txt", "w") as output:
        output.write(str(list_train_metric_logs))

    with open("full_valid_curve_14.05_500_40epoch_lr_10.txt", "w") as output:
        output.write(str(list_valid_logs))

    with open("full_valid_metric_curve_14.05_500_40epoch_lr_10.txt", "w") as output:
        output.write(str(list_valid_metric_logs))

    """
    plt.figure(dpi=300)
    plt.plot(train_curve, label='loss', linewidth=0.5)
    plt.plot(metric_curve, label='IOU_score', linewidth=0.5)
    plt.legend()
    plt.xlabel('batch')
    plt.show()
    """

torch.save(model, './best_model_14_05_end_500_40epoch_lr_10.pth')


# plt.figure(dpi=300)
# plt.plot(list_train_curve, label='loss', linewidth=0.5)
# plt.plot(list_metric_curve, label='IOU_score', linewidth=0.5)
# plt.legend()
# plt.xlabel('epoch')
# plt.show()


# with open("train_curve_08.05_500epoch_augmentations.txt", "w") as output:
#     output.write(str(list_train_curve))
#
# with open("metric_curve_08.05_500epoch_augmentations.txt", "w") as output:
#     output.write(str(list_metric_curve))


#print("METRIC",np.array(list_all_metric_logs).flatten())
#print("LOSS", list_all_train_logs)

# list_all_train_logs = np.array(list_all_train_logs).flatten()
# list_all_metric_logs = np.array(list_all_metric_logs).flatten()

plt.figure(dpi=300)
plt.plot(list_train_logs, label='loss', linewidth=0.5)
plt.plot(list_train_metric_logs, label='IOU_score', linewidth=0.5)
plt.legend()
plt.xlabel('epoch')
plt.show()


plt.figure(dpi=300)
plt.plot(list_valid_logs, label='loss', linewidth=0.5)
plt.plot(list_valid_metric_logs, label='IOU_score', linewidth=0.5)
plt.legend()
plt.xlabel('epoch')
plt.show()


#np.set_printoptions(threshold = np.prod(list_all_train_logs.shape))

# with open("full_train_curve_14.05_500epoch_augmentations.txt", "w") as output:
#     output.write(str(list_all_train_logs))
#
# with open("full_metric_curve_14.05_500epoch_augmentations.txt", "w") as output:
#     output.write(str(list_all_metric_logs))
#