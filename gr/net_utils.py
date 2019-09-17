#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Deep-learning network common functions
#
# Author:      kol
#
# Created:     19.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

import sys
if sys.version_info[0] < 3:
    from grdef import *
else:
    from gr.grdef import *
    from gr.utils import format_stone_pos
import cv2
import string as ss

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def show_detections(im, class_name, dets, thresh=0.5, f_label = True, f_title = True, title = None, ptype = "rect"):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print("No predictions for class {}".format(class_name))
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        if ptype == "rect" or ptype == "r":
            ax.add_patch(
                plt.Rectangle((int(bbox[0]), int(bbox[1])),
                              int(bbox[2] - bbox[0]),
                              int(bbox[3] - bbox[1]), fill=False,
                              edgecolor='red', linewidth=3)
                )
        elif ptype == "circle" or ptype == "c":
            ax.add_patch(
                plt.Circle((int(bbox[0]), int(bbox[1])),
                              int(max(bbox[2] - bbox[0], bbox[3] - bbox[1])),
                              fill=False,
                              edgecolor='red', linewidth=3)
                )
        if f_label:
            ax.text(int(bbox[0]), int(bbox[1] - 2),
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    if f_title:
        if title is None:
          title = '{} detections with p({} | box) >= {:.1f}'.format(class_name, class_name, thresh)

        ax.set_title(title, fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def cv2_show_detections(im, class_name, dets, thresh=0.5, f_label = True, color = (0,0,255), ptype = "rect"):
    """Draw detected bounding boxes on cv2 image"""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print("No predictions for class {}".format(class_name))
        return False

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        if ptype == "rect" or ptype == "r":
            cv2.rectangle(im, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 1)
        elif ptype == "circle" or ptype == "c":
            cv2.circle(im, (int(bbox[0]), int(bbox[1])),
                              int(max(bbox[2] - bbox[0], bbox[3] - bbox[1])),
                              color, 1)
        if f_label:
            cv2.putText(im,
                    '{:s} {:.3f}'.format(class_name, score),
                    (int(bbox[0]), int(bbox[1] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)


def make_anno(filename, img_file, img, jgf = None):
    pass
