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
#from samples.coco.coco import CocoConfig
#from mrcnn import model as modellib
#from mrcnn import visualize

# Reading gaze data

header_raw = pd.read_csv('data/gaze_synchrone.csv', sep=' ', header=0, skiprows=2, nrows=0)
gaze_data = pd.read_csv('data/gaze_synchrone.csv', sep=' ', header=0, skiprows=3)

header_raw = header_raw.columns[1:33]

gaze_data.columns = header_raw

# Sorting columns needed for mapping points
gaze_coordinates = gaze_data[['right_eye.gaze_point.position_on_display_area.x()', 'right_eye.gaze_point.position_on_display_area.y()', 'left_eye.gaze_point.position_on_display_area.x()', 'left_eye.gaze_point.position_on_display_area.y()']]

gaze_coordinates.columns = ['right_x', 'right_y', 'left_x', 'left_y']
gaze_coordinates = gaze_coordinates.replace('-nan(ind)', '')
gaze_coordinates = gaze_coordinates.replace(r'^\s*$', np.nan, regex=True)
gaze_coordinates = gaze_coordinates.to_numpy().astype(float)

# Get image size

temp_image = PIL.Image.open(r'data\Frames\frame000501.png')
(img_width, img_height) = temp_image.size
print ('image size:',(img_width, img_height))
print(type(img_width))
gaze_coordinates[:,[1,3]] *= img_height
gaze_coordinates[:,[0,2]] *= img_width


# Read images

for img_no in range(500, 510):
    img = mpimg.imread(r'data\Frames\frame000505.png')

    x = [gaze_coordinates[img_no,0], gaze_coordinates[img_no,2]]
    y = [gaze_coordinates[img_no,1], gaze_coordinates[img_no,3]]
    s = 100 - 20*abs(img_no - 505)
    plt.scatter(x, y, s=s, color = 'blue')
print(np.sum(img[:,:,2]==1.0))
plt.imshow(img)
plt.show()

"""
def prepare_mrcnn_model(model_path, model_name, class_names, my_config):
    classes = open(class_names).read().strip().split("\n")
    print("No. of classes", len(classes))

    hsv = [(i / len(classes), 1, 1.0) for i in range(len(classes))]
    COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(42)
    random.shuffle(COLORS)

    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=my_config)
    model.load_weights(model_name, by_name=True)

    return COLORS, model, classes

def custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation):
    detections = model.detect([test_image], verbose=1)[0]

    if mrcnn_visualize:
        matplotlib.use('TkAgg')
        visualize.display_instances(test_image, detections['rois'], detections['masks'], detections['class_ids'], classes, detections['scores'])
        return

    if instance_segmentation:
        hsv = [(i / len(detections['rois']), 1, 1.0) for i in range(len(detections['rois']))]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(42)
        random.shuffle(colors)

    for i in range(0, detections["rois"].shape[0]):
        classID = detections["class_ids"][i]

        mask = detections["masks"][:, :, i]
        if instance_segmentation:
            color = colors[i][::-1]
        else:
            color = colors[classID][::-1]

        # To visualize the pixel-wise mask of the object
        test_image = visualize.apply_mask(test_image, mask, color, alpha=0.5)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    if draw_bbox:
        for i in range(0, len(detections["scores"])):
            (startY, startX, endY, endX) = detections["rois"][i]

            classID = detections["class_ids"][i]
            label = classes[classID]
            score = detections["scores"][i]

            if instance_segmentation:
                color = [int(c) for c in np.array(colors[i]) * 255]

            else:
                color = [int(c) for c in np.array(colors[classID]) * 255]

            cv2.rectangle(test_image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.2f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(test_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return test_image

def perform_inference_image(image_path, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation, save_enable):
    test_image = cv2.imread(image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    output = custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation)
    if not mrcnn_visualize:
        if save_enable:
            cv2.imwrite("result.png", output)
        cv2.imshow("Output", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
"""