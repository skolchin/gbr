#-------------------------------------------------------------------------------
# Name:        Go board parsing project
# Purpose:     Misc functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import grdef
import cv2
import numpy as np
from PIL import Image, ImageTk

# Show image
# Simple wrapper on cv2.imshow
def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()

# Make image displaying stones
# The function takes image shape and array of stone coordinates (X,Y,R)
# The function creates a new image with the same shape and draw the stones there
def make_stones_img(shape, points, color = grdef.COLOR_BLACK, img = None):
    if (img is None):
       img = np.full(shape, grdef.COLOR_WHITE[0], dtype=np.uint8)

    for i in points:
        x1 = i[0]
        y1 = i[1]
        r = 7
        cv2.circle(img, (x1,y1), r, color, -1)

    return img

# Show stones
# The function takes image shape and array of stone coordinates (X,Y,R)
# The function creates a new image with the same shape and draw the stones there
# If f_show = TRUE, image is been shown
def show_stones(title, shape, points, img = None):
    img = make_stones_img(shape, points, img)
    show(title, img)

# Make image displaying given lines
# If an image is provided, lines are drawn on it
# Otherwise it creates a new image with the same shape and draw the lines there
def make_lines_img(shape, lines, width = 1, img = None):
    if (img is None):
       img = np.full(shape, grdef.COLOR_WHITE[0], dtype=np.uint8)
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        cv2.line(img, (x1,y1), (x2,y2), grdef.COLOR_BLACK, width)

    return img

# Show lines
# The function takes image shape and array of line coordinates (X1,Y1,X2,Y2)
# If an image is provided, lines are drawn on it
# Otherwise it creates a new image with the same shape and draw the lines there
def show_lines(title, shape, lines, img = None):
    img = make_lines_img(shape, lines, img)
    show(title, img)

# Convert CV2 image to Tkinter format
def img_to_imgtk(img):
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    return imgtk

# Utility function - elmininates duplicates in an array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# Remove values too close to each other
def remove_nearest(a, axis1 = 0, axis2 = None, delta = 5):
    if a is None or len(a) == 0:
       return a

    # Sort the array
    b = None
    if type(a[0]) is not tuple:
       b = sorted(a)
    else:
       if axis2 is None:
          b = sorted(a, key = lambda x: x[axis1])
       else:
          b = sorted(a, key = lambda x: x[axis1][axis2])

    def get_v(x):
        if type(x) is not tuple:
           return x
        else:
            if axis2 is None:
                return x[axis1]
            else:
                return x[axis1][axis2]


    # Find values not too close and add to returning array
    r = []
    vp = None
    for i in b:
        v = get_v(i)
        if vp is None or abs(v - vp) > delta:
           r.append(i)
        vp = v

    return r

# Convert 1-channel image to 3-channel
def img1_to_img3(img):
    img3 = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img3[:,:,0] = img
    img3[:,:,1] = img
    img3[:,:,2] = img

    return img3

# Check horizontal and vertical lines are intersecting
def has_intersection(lh, lv):
    min_y = min(lv[0][1], lv[1][1])
    max_y = max(lv[0][1], lv[1][1])
    y = lh[0][1]
    f = min_y <= y and max_y >= y
    return f

# Remove lines too close to board edges
def clear_lines(shape, lines, delta = 3):
    res = []
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        if (x1 > delta and y1 > delta and x2 < shape[0] - delta and y2 < shape[1] - delta):
           res.append(i)

    return res

# Transform HoughP lines results to array with coordinate tuples
# HoughP returns lines as array (N, 4)
def houghp_to_lines(lines):
    res = []
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        res.append(((x1,y1),(x2,y2)))

    return res

# Transform HoughP lines results to array with coordinate tuples
# Hough returns rho and theta but not actual coordinates
def hough_to_lines(lines, shape):
    res = []
    for i in lines:
        rho,theta = i[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        #if a != 0 and b != 0:
        x1 = int(x0 + shape[0]*(-b))
        y1 = int(y0 + shape[1]*(a))
        x2 = int(x0 - shape[0]*(-b))
        y2 = int(y0 - shape[1]*(a))

        res.append(((x1,y1),(x2,y2)))

    return res
