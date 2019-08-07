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

CLASSES = ["black", "white"]
COLORS = [(0,0,0),(255,255,255)]
MAX_CONF = 0.5

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
rcnn_path = root_path.joinpath('py-faster-rcnn')

model_file = str(rcnn_path.joinpath("models\\gbr\\test.prototxt"))
weigth_file = str(rcnn_path.joinpath("output\\faster_rcnn_end2end\\train\\gf_zf_faster_rcnn_iter_100.caffemodel"))
img_file = str(root_path.joinpath("img\\go_board_13_gen.png"))

sys.path.append(str(rcnn_path.joinpath('lib')))

import fast_rcnn.config as cfg
cfg.cfg_from_file(str(rcnn_path.joinpath("models\\gbr\\faster_rcnn_end2end.yml")))

caffe.set_mode_cpu()

net = caffe.Net(model_file, weigth_file, caffe.TEST)

print("== Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

print("== Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

print("== Inputs:")
print(net.inputs)

img = caffe.io.load_image(img_file)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2,1,0))
tr_img = transformer.preprocess('data', img)

net.blobs['data'].data[...] = tr_img
detections = net.forward()

print("== Detections:")
for key in detections:
    print("{}: {}".format(key, detections[key]))

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
