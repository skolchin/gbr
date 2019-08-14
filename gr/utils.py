#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Misc functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

from gr.grdef import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import string as ss

# Show image
# Simple wrapper on cv2.imshow
def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()

# Make image displaying stones
# The function takes image shape and array of stone coordinates (X,Y,R)
# The function creates a new image with the same shape and draw the stones there
def make_stones_img(shape, points, color = COLOR_BLACK, img = None):
    if (img is None):
       img = np.full(shape, COLOR_WHITE[0], dtype=np.uint8)

    for i in points:
        x1 = i[GR_X]
        y1 = i[GR_Y]
        r = 7
        cv2.circle(img, (x1,y1), r, color, -1)

    return img

# Show stones
# The function takes image shape and array of stone coordinates (X,Y,R)
# The function creates a new image with the same shape and draw the stones there
# If f_show = TRUE, image is been shown
def show_stones(title, shape, points, img = None):
    if points is None:
       return

    img = make_stones_img(shape, points, img)
    show(title, img)

# Make image displaying given lines
# If an image is provided, lines are drawn on it
# Otherwise it creates a new image with the same shape and draw the lines there
def make_lines_img(shape, lines, width = 1, img = None):
    if (img is None):
       img = np.full(shape, COLOR_WHITE[0], dtype=np.uint8)
    for i in lines:
        x1 = i[GR_FROM][GR_X]
        y1 = i[GR_FROM][GR_Y]
        x2 = i[GR_TO][GR_X]
        y2 = i[GR_TO][GR_Y]
        cv2.line(img, (x1,y1), (x2,y2), COLOR_BLACK, width)

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

    # Subfunction for tuples
    def remove_nearest_t(a, axis1, axis2, delta):
        b = None
        if axis2 is None: b = sorted(a, key = lambda x: x[axis1])
        else:             b = sorted(a, key = lambda x: x[axis1][axis2])
        r = []
        vp = None
        for i in b:
            v = None
            if axis2 is None: v = i[axis1]
            else:             v = i[axis1][axis2]
            if vp is None or abs(v - vp) > delta: r.append(i)
            vp = v
        return r

    # Subfunction for ndarrays
    def remove_nearest_a(a, axis1, axis2, delta):
        b = np.sort(a, axis1)
        r = []
        vp = [None, None]
        vf = np.vectorize(lambda t: t > delta)

        for i in b:
            v = None
            if axis2 is None:
               v = [i[axis1], None]
            else:
               v = [i[axis1], i[axis2]]
            if vp[0] is None: r.append(i)
            else:
                t = np.abs(np.subtract(v, vp))
                ft = vf(t)
                if ft.any(): r.append(i)

            vp = v
        return np.asarray(r)

    # Subfunction for other types
    def remove_nearest_r(a, axis1, axis2, delta):
        b = sorted(a)
        r = []
        vp = None
        for i in b:
            v = i
            if vp is None or abs(v - vp) > delta: r.append(i)
            vp = v
        return r

    if a is None or len(a) == 0:
       return a
    elif type(a[0]) is tuple:
       return remove_nearest_t(a, axis1, axis2, delta)
    elif type(a) is np.ndarray:
       return remove_nearest_a(a, axis1, axis2, delta)
    else:
       return remove_nearest_r(a, axis1, axis2, delta)

# Convert 1-channel image to 3-channel
def img1_to_img3(img):
    img3 = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img3[:,:,0] = img
    img3[:,:,1] = img
    img3[:,:,2] = img

    return img3

# Check horizontal and vertical lines are intersecting
def has_intersection(lh, lv):
    min_y = min(lv[GR_FROM][GR_Y], lv[GR_TO][GR_Y])
    max_y = max(lv[GR_FROM][GR_Y], lv[GR_TO][GR_Y])
    y = lh[GR_FROM][GR_Y]
    f = min_y <= y and max_y >= y
    return f

# Remove lines too close to board edges
def clear_lines(shape, lines, delta = 3):
    res = []
    for i in lines:
        x1 = i[GR_FROM][GR_X]
        y1 = i[GR_FROM][GR_Y]
        x2 = i[GR_TO][GR_X]
        y2 = i[GR_TO][GR_Y]
        if (x1 > delta
           and y1 > delta
           and x2 < shape[CV_WIDTH] - delta
           and y2 < shape[CV_HEIGTH] - delta):
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

def stone_pos(stone, axis = None):
    if axis is None:
         return ss.ascii_uppercase[stone[GR_A]-1] + str(stone[GR_B])
    elif axis == GR_A:
         return ss.ascii_uppercase[stone[GR_A]-1]
    elif axis == GR_B:
         return str(stone[axis])
    else:
         return int(round(stone[axis],0))

# Convert GR results to JGF dictionary
def gres_to_jgf(res):

    def sp(stones):
        p = dict()
        for stone in stones:
            key = stone_pos(stone)
            p[key] = dict()
            p[key]['X'] = stone_pos(stone, GR_X)
            p[key]['Y'] = stone_pos(stone, GR_Y)
            p[key]['R'] = stone_pos(stone, GR_R)
            p[key]['A'] = stone_pos(stone, GR_A)
            p[key]['B'] = stone_pos(stone, GR_B)
        return p

    jgf = dict()
    jgf['board_size'] = res[GR_BOARD_SIZE]
    jgf['edges'] = dict()
    jgf['edges']['0'] = res[GR_EDGES][0]
    jgf['edges']['1'] = res[GR_EDGES][1]
    jgf['spacing'] = res[GR_SPACING]
    jgf['num_stones'] = (len(res[GR_STONES_B]), len(res[GR_STONES_W]))
    jgf['black'] = sp(res[GR_STONES_B])
    jgf['white'] = sp(res[GR_STONES_W])
    return jgf

# Resize the image proportionally so no side will exceed given max_size
# if f_upsize = False, images with size less than max_size are not upscaled
def resize(img, max_size, f_upsize = True):
    im_size_max = np.max(img.shape[0:2])
    im_size_min = np.min(img.shape[0:2])
    im_scale = float(max_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    if not f_upsize and im_scale > 1.0:
        return img
    else:
        return cv2.resize(img, dsize = None, fx = im_scale, fy = im_scale)

