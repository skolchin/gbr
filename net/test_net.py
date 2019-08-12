#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      skolchin
#
# Created:     19.07.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
import cv2
#import cv2.dnn as dnn
import caffe
import numpy as np
from matplotlib import pyplot as plt
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

def vis_detections(im, class_name, dets, thresh=0.5):
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

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()



CLASSES = ["_back_", "white", "black"]

#net = dnn.readNetFromCaffe(ROOT + "\\net\\gbr.prototxt", ROOT + "\\net\\gbr.caffemodel")

#img = cv2.imread(ROOT + "\\img\\go_board_13_gen.png")
#if img is None:
#    raise Exception("Invalid image")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.resize(img, (300,300))
#blob = dnn.blobFromImage(img)
#net.setInput(blob)

#for i in range(len(net.getLayerNames())+1):
#    l = net.getLayer(i)
#    print(l.name, l.type, len(l.blobs))

#detections = net.forward()

root_path = Path(__file__).with_name('').joinpath('..').resolve()
print("Root path is {}".format(root_path))
rcnn_path = root_path.joinpath('..','py-faster-rcnn').resolve()
print("RCNN path is {}".format(rcnn_path))

model_file = str(root_path.joinpath("net", "models","test.prototxt"))
print("Model file is {}".format(model_file))
weigth_file = str(root_path.joinpath("net", "models","out", "gbr_zf", "train", "gbr_zf_iter_20000.caffemodel"))
print("Using weights {}".format(weigth_file))
img_file = str(root_path.joinpath("img","go_board_5.jpg"))
print("Image {}".format(img_file))

cfg.TEST.HAS_RPN = True
caffe.set_mode_gpu()

net = caffe.Net(model_file, weigth_file, caffe.TEST)

print("== Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

print("== Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

##img = caffe.io.load_image(img_file)
##transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
##transformer.set_transpose('data', (2,0,1))
##transformer.set_raw_scale('data', 255.0)
##transformer.set_channel_swap('data', (2,1,0))
##tr_img = transformer.preprocess('data', img)
##
##net.blobs['data'].data[...] = tr_img
##detections = net.forward()
##
##print("== Detections:")
##for key in detections:
##    print("{}: {}".format(key, detections[key]))

img = cv2.imread(img_file)
scores, boxes = im_detect(net, img)

print("== Detections")
print("Scores: {}".format(scores))
print("Boxes: {}".format(boxes))

CONF_THRESH = 0.8
NMS_THRESH = 0.3
for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    vis_detections(img, cls, dets, thresh=CONF_THRESH)

plt.show()

# loop over the detections
##for i in np.arange(0, detections.shape[2]):
##    confidence = detections[0, 0, i, 2]
##    print("Confidence:", confidence)
##    if confidence > MAX_CONF:
##        idx = int(detections[0, 0, i, 1])
##        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
##        (startX, startY, endX, endY) = box.astype("int")
##
##        # display the prediction
##        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
##        print("[INFO] {}".format(label))
##        cv2.rectangle(img, (startX, startY), (endX, endY),
##                COLORS[idx], 2)
##        y = startY - 15 if startY - 15 > 15 else startY + 15
##        cv2.putText(img, label, (startX, y),
##                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
##
### show the output image
##cv2.imshow("Output", img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
