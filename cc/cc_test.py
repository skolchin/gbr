#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     22.12.2019
# Copyright:   (c) kol 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2

img = cv2.imread("../img/go_board_2.png")
if img is None:
    raise Exception("Not found")

cascade = cv2.CascadeClassifier("m/cascade.xml")
results = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
maxSize=(30, 30))
results[:,2:] += results[:,:2]
print(results)
for r in results:
    cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), color = (0, 0, 255))

cv2.imshow('Results', img)
cv2.waitKey()
cv2.destroyAllWindows()
