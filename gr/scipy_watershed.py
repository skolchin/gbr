#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Watershed function (SciPy version)
#
# Author:      skolchin
#
# Created:     20.08.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import cv2
import numpy as np
from skimage.morphology import watershed
from scipy import ndimage

# Apply watershed transformation
# The algorithm was originally developed by Adrian Rosebrock (PyImageSearch.com)
# and adopted so it currently uses stone coordinations as an indicators of peaks
# instead of original "max peak value" method
# TODO: rewrite to use only OpenCV methods
def apply_watershed(img, stones, n_thresh, f_bw, n_morph = 0, f_debug = False):
    WS_MAX = 255

    kernel  = np.ones((3,3),np.uint8)
    if f_bw == 'B':
       # To have watershed properly determine black stones, board dividers
       # have to be removed with dilation, source image converted to negative
       img2 = cv2.dilate(img,kernel,iterations=n_morph+1)
       img2 = cv2.erode(img2,kernel,iterations=n_morph+1)
       img2 = cv2.bitwise_not(img2)
       ret, t2 = cv2.threshold(img2, n_thresh, WS_MAX, cv2.THRESH_BINARY)
    else:
       # White stones, normal thresholding
       if n_morph > 0:
            img2 = cv2.erode(img, kernel, iterations = n_morph)
       else:
            img2 = img
       ret, t2 = cv2.threshold(img2, n_thresh, WS_MAX, cv2.THRESH_BINARY)

    if f_debug:
       cv2.imshow('Source', img2)
       cv2.imshow('Threshold', t2)

    # Apply distance transformation
    D = ndimage.distance_transform_edt(t2)
    if f_debug:
       d = D.copy()
       cv2.normalize(d, d, 0, 1.0, cv2.NORM_MINMAX)
       cv2.imshow('Distance', d)

    # Use stones coordinations as peaks
    peaks = np.zeros(t2.shape[:2], dtype = np.bool)
    for st in stones:
        x = int(st[0])
        y = int(st[1])
        peaks[y,x] = True

    # Set the markers and apply watershed
    markers = ndimage.label(peaks, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=t2)

    # Collect results
    dst = np.zeros(img.shape, dtype=np.uint8)
    rt = []
    for label in np.unique(labels):
        if label == 0: continue

        mask = np.zeros(t2.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r < 20:
           rt.append ([int(x), int(y), int(r)])
           cv2.circle(dst, (int(x),int(y)), int(r), (255,255,255), -1)

    dst = cv2.bitwise_not(dst)
    if f_debug: cv2.imshow('Result', dst)

    return np.array(rt), dst
