#-------------------------------------------------------------------------------
# Name:        Test watershed module
# Purpose:
#
# Author:      skolchin
#
# Created:     19.07.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import numpy as np
from gr.board import GrBoard
import random as rng
from gr.cv2_watershed import apply_watershed

rng.seed(12345)

def process(f_bw):

    #img = cv2.imread("img\\go_board_1.png")
    board = GrBoard("img\\go_board_44.png")
    img = board.image
    if img is None: raise Excetion("None")
    cv2.imshow('Original', img)

    #pm_img = cv2.pyrMeanShiftFiltering(img, 21, 51)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray', gray)

    b,g,r = cv2.split(img)
    if f_bw == 'B':
       gray = r
    else:
       gray = b

##    for i in range(200):
##        n = i+50
##        print(n)
##        ret, dst = apply_watershed(gray, board.stones[f_bw], n, f_bw, True)
##        cv2.waitKey()
##    return

    #kernel = np.ones((3, 3), dtype = np.uint8)
    #gray = cv2.dilate(gray, kernel, iterations = 3)

    ret, dst = apply_watershed(gray, board.stones[f_bw], 140, f_bw, 4, True)
    print("{} of {} stones found".format(len(ret), len(board.stones[f_bw])))

    # Generate random colors
    colors = []
    for i in ret:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))

    # Create the result image
    dst = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(ret)):
        x = ret[i, 0]
        y = ret[i, 1]
        r = ret[i, 2]
        cv2.circle(dst, (x, y), r, colors[i], -1)

    # Visualize the final image
    cv2.imshow('Final result', dst)


process('B')

cv2.waitKey()
cv2.destroyAllWindows()


