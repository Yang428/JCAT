import vot2020 as vot
import torch
import numpy as np
import sys
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('/home/wcz/Yang/JCAT/')

from pytracking.tracker.Jcat_mask import Jcat
from pytracking.parameter.Jcat import Jcat_vot20 as vot_params
from ltr.data.bounding_box_utils import masks_to_bboxes

def rect_to_poly(rect):
    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[0] + rect[2]
    y1 = rect[1]
    x2 = rect[0] + rect[2]
    y2 = rect[1] + rect[3]
    x3 = rect[0]
    y3 = rect[1] + rect[3]
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def parse_sequence_name(image_path):
    idx = image_path.find('/color/')
    return image_path[idx - image_path[:idx][::-1].find('/'):idx], idx

def parse_frame_name(image_path, idx):
    frame_name = image_path[idx + len('/color/'):]
    return frame_name[:frame_name.find('.')]

# MAIN
handle = vot.VOT("mask")
vot_anno = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

params = vot_params.parameters()

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

vot_anno_mask = vot.make_full_size(vot_anno, (image.shape[1], image.shape[0]))
bbox = masks_to_bboxes(torch.from_numpy(vot_anno_mask), fmt='t').squeeze().tolist()

sequence_name, idx_ = parse_sequence_name(imagefile)
frame_name = parse_frame_name(imagefile, idx_)

params.masks_save_path = ''
params.save_mask = False

tracker = Jcat(params)

# tell the sequence name to the tracker (to save segmentation masks to the disk)
tracker.sequence_name = sequence_name
tracker.frame_name = frame_name

tracker.initialize(image, bbox, vot_anno_mask)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    # tell the frame name to the tracker (to save segmentation masks to the disk)
    frame_name = parse_frame_name(imagefile, idx_)
    tracker.frame_name = frame_name

    pred_mask = tracker.track(image).astype(np.uint8)

    handle.report(pred_mask, 1.0)
