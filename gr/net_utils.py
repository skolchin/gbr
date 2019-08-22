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

import sys
if sys.version_info[0] < 3:
    from grdef import *
else:
    from gr.grdef import *
import cv2

import cv2
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


def make_anno(meta_file, image_file, img = None, jgf = None):

    def annotate_stones(file, jgf, shape, cls):
        stones = jgf[cls]
        if stones is None:
           return
        bbox = np.empty((len(stones),5), dtype = np.float)

        # Find radius which appears most often
        rlist = [f[1]['R'] for f in stones.items()]
        unique, counts = np.unique(rlist, return_counts=True)
        summary = dict(zip(unique, counts))
        max_r = max(summary, key = lambda x: summary[x])

        # Proceed
        n = 0
        for i in stones:
            x = stones[i]['X']
            y = stones[i]['Y']
            r = stones[i]['R']
            r = max_r

            a = r + 2
            xmin = x - int(a)
            if xmin <= 0: xmin = 1
            ymin = y - int(a)
            if ymin <= 0: ymin = 1
            xmax = x + int(a)
            if xmax > shape[1]: xmax = shape[1]
            ymax = y + int(a)
            if ymax > shape[0]: ymax = shape[0]

            bbox[n,0] = xmin
            bbox[n,1] = ymin
            bbox[n,2] = xmax
            bbox[n,3] = ymax
            bbox[n,4] = 100.0
            n += 1

            line = '\n\t<object>'
            line += '\n\t\t<name>' + cls + '</name>\n\t\t<pose>Unspecified</pose>'
            line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>Unspecified</difficult>'

            line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
            line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
            line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
            line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
            line += '\n\t\t</bndbox>'

            line += '\n\t</object>'

            file.write(line)

        return bbox

    f = open(str(meta_file),'w')

    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t<filename>' + Path(image_file).name + '</filename>' + '\n'
    f.write(line)
    line = '\t<path>' + str(Path(image_file)) + '</path>' + '\n'
    f.write(line)
    line = '\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
    f.write(line)

    if img is None:
       img = cv2.imread(str(image_file))
       if img is None: raise Exception("File not found {}".format(image_file))

    (height, width, depth) = img.shape
    line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
    line += '\t<depth>' + str(depth) + '</depth>\n\t</size>'
    f.write(line)
    line = '\n\t<segmented>Unspecified</segmented>'
    f.write(line)

    bb_b = None
    bb_w = None
    if not jgf is None:
        bb_b = annotate_stones(f, jgf, (height, width), 'black')
        bb_w = annotate_stones(f, jgf, (height, width), 'white')

    line = "\n</annotation>" + '\n'
    f.write(line)
    f.close()

    return bb_b, bb_w

