#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Deep-learning network common functions
#
# Author:      skolchin
#
# Created:     19.07.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt

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
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3)
                )
        elif ptype == "circle" or ptype == "c":
            ax.add_patch(
                plt.Circle((bbox[0], bbox[1]),
                              max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                              fill=False,
                              edgecolor='red', linewidth=3)
                )
        if f_label:
            ax.text(bbox[0], bbox[1] - 2,
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

