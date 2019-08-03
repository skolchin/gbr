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

import cv2
import cv2.dnn as dnn
import numpy as np
from matplotlib import pyplot as plt

CLASSES = ["black", "white"]
COLORS = [(0,0,0),(255,255,255)]
MAX_CONF = 0.5

net = dnn.readNetFromCaffe("C:\\Users\\skolchin\\Documents\\kol\\gbr\\net\\ssd.prototxt",
                           "C:\\Users\\skolchin\\Documents\\kol\\gbr\\net\\ssd.caffemodel")

img = cv2.imread("C:\\Users\\skolchin\\Documents\\kol\\gbr\\img\\go_board_1.png")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (300,300))

blob = dnn.blobFromImage(img, size=(300, 300), crop = False)
net.setInput(blob)

for i in range(len(net.getLayerNames())+1):
    l = net.getLayer(i)
    print(l.name, l.type, len(l.blobs))

detections = net.forward()
print("Detections:", detections.shape)

# loop over the detections
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    print("Confidence:", confidence)
    if confidence > MAX_CONF:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        cv2.rectangle(img, (startX, startY), (endX, endY),
                COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(img, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
