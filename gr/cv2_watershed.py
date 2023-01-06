# Go board recognition project
# Watershed function (OpenCV version)
# (c) kol, 2019-2023

import cv2
import numpy as np
import logging

# Apply watershed transformation
# The algorithm derived from OpenCV's publication 'Image Segmentation with Watershed Algorithm'
# and adopted to use stone coordinations as an indicators of peaks instead of original "max peak value" method
def apply_watershed(gray, stones, n_thresh, f_bw, n_morph = 0, f_debug = False):
    """Apply watershed transformation to given board image.

    gray        source image (either gray or one of channels)
    stones      array of board stone X,Y coordinations determined by some other method
    n_thresh    threshold level
    f_bw        either B for black stones or W for white
    n_morph     number of iterations of morphological transformation (0 if not needed)
    f_debug     if True, debug images are to be shown with cv2.imshow() call

    Returns     array of stones in X,Y,R format and a debug image with stones plotted
    """

    # Preprocess image
    kernel = np.ones((3, 3), dtype = np.uint8)
    if f_bw == 'B':
       # To have watershed properly determine black stones, board dividers have to be removed
       # Therefore, dilate() is applied n_morph+1 times
       n_morph += 1
       gray = cv2.dilate(gray, kernel, iterations = n_morph)

       # Converted source image to negative
       gray = cv2.bitwise_not(gray)
    elif n_morph > 0:
       # White images should be eroded, not dilated
       gray = cv2.erode(gray, kernel, iterations = n_morph)

    if f_debug:
       cv2.imshow('Gray', gray)

    # Prepare thresholded image
    _, thresh = cv2.threshold(gray, n_thresh, 255, cv2.THRESH_BINARY)
    if f_debug: cv2.imshow('Threshold', thresh)

    #b = np.zeros(stones.shape, dtype = np.bool)
    #b[7] = 1
    #stones = stones[b].reshape(1, stones.shape[1])

    # Prepare peaks map
    peaks = np.zeros(thresh.shape, dtype = np.uint8)
    for i in range(len(stones)):
        x = int(stones[i,0])
        y = int(stones[i,1])

        if x >= 0 and y >= 0 and x < thresh.shape[1] and y < thresh.shape[0] and thresh[y,x] > 0:
           #cv2.circle(peaks, (x,y), 1, (i+1), -1)
           peaks[y,x] = (i+1)
        else:
           # Circle center falls to a black point
           # Look around to find a white point within small radius
           r = 5
           r2 = r*2
           f = False
           for dy in range(r2+1):
               for dx in range(r2+1):
                   vy = y+r-dy
                   vx = x+r-dx
                   if vx >= 0 and vy >= 0 and vx < thresh.shape[1] and vy < thresh.shape[0] and thresh[vy, vx] > 0:
                      f = True
                      #cv2.circle(peaks, (x+r2-dx,y+r2-dy), 1, (i+1), -1)
                      peaks[vy, vx] = (i+1)
                      break
               if f: break
           if not f:
              # No white found. Ignore the stone, but save a warning
              logging.warning('WATERSHED: Cannot find peak for stone ({},{},{})'.format(x,y,r))

    if f_debug:
       m = np.zeros((thresh.shape[0],thresh.shape[1],3), dtype = np.uint8)
       for i in range(3): m[:,:,i] = thresh
       for y in range(peaks.shape[0]):
           for x in range(peaks.shape[1]):
               if peaks[y,x] > 0:
                  for i in range(3):
                      for j in range(3):
                          m[y+i-1,x+j-1] = (0,0,255)
       cv2.imshow('Peaks', m)

##    # Do distance transform
##    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
##    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
##    if f_debug: cv2.imshow('Distance', dist)
##
##    # Threshold to normalize DT to 1
##    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
##
##    # Find total markers
##    dist_8u = peaks.astype('uint8')
##    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
##    # Create the marker image for the watershed algorithm
##    markers = np.zeros(thresh.shape, dtype=np.int32)
##
##    # Draw the foreground markers
##    for i in range(len(contours)):
##        cv2.drawContours(markers, contours, i, (i+1), -1)
##
##    # Draw the background marker
##    cv2.circle(markers, (5,5), 3, (255,255,255), -1)
##
##    if f_debug:
##       m = np.array(markers*10000).astype(np.uint8)
##       cv2.imshow('Markers', m)
##
##    # Apply watershed
##    img3 = np.empty((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
##    img3[:,:,0] = gray
##    img3[:,:,1] = gray
##    img3[:,:,2] = gray
##    cv2.watershed(img3, markers)

    # Apply watershed
    markers = peaks.astype(np.int32)
    img3 = np.empty((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(3): img3[:,:,i] = thresh
    cv2.watershed(img3, markers)

    if f_debug:
       mark = markers.astype('uint8')
       #mark = cv2.bitwise_not(mark)
       cv2.imshow('Markers_v2', mark)
       m = img3.copy()
       m[markers == -1] = [0,0,255]
       cv2.imshow('Borders', m)

    # Collect results
    dst = np.zeros(gray.shape, dtype=np.uint8)
    rt = []
    for c in np.unique(markers):
        if c <= 0: continue
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == c] = 255
        if cv2.__version__.startswith('3'):
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        else:
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cm = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(cm)
        if f_debug: logging.info("CV2_WATERSHED: marker {}: ({}, {}, {})".format(c,x,y,r))

        if r > 20.0:
            logging.info("WATERSHED: Ignoring marker {}: ({}, {}, {})".format(c,x,y,r))
        else:
           # Increase radius to number of pixels removed with erode/dilate
           rt.append ([int(x), int(y), int(r + n_morph)])
           cv2.circle(dst, (int(x), int(y)), int(r), (255,255,255), -1)

    # Filter out outlied R's
##    if len(rt) > 0:
##        rlist = [f[2] for f in rt]
##        mean_r = sum(rlist) / float(len(rlist))
##        rt2 = []
##        for r in rt:
##            if r[2] >= mean_r-3 and r[2] <= mean_r+3:
##                rt2.append(r)
##                cv2.circle(dst, (r[0],r[1]), r[2], (255,255,255), -1)
##
##        rt = rt2

    dst = cv2.bitwise_not(dst)
    if f_debug: cv2.imshow('Result', dst)

    return np.array(rt), dst
