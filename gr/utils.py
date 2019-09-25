#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Misc functions
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

import sys
if sys.version_info[0] < 3:
    from grdef import *
else:
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
        r = i[GR_R]
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
def make_lines_img(shape, lines, width = 1, color = COLOR_BLACK, img = None):
    if (img is None):
       img = np.full(shape, COLOR_WHITE[0], dtype=np.uint8)
    for i in lines:
        x1 = i[GR_FROM][GR_X]
        y1 = i[GR_FROM][GR_Y]
        x2 = i[GR_TO][GR_X]
        y2 = i[GR_TO][GR_Y]
        cv2.line(img, (x1,y1), (x2,y2), color, width)

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
    if len(img.shape) ==3 and img.shape[2] == 3:
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
    for i in range(3): img3[:,:,i] = img
    return img3

# Check horizontal and vertical lines are intersecting
def has_intersection(lh, lv):
    min_y = min(lv[GR_FROM][GR_Y], lv[GR_TO][GR_Y])
    max_y = max(lv[GR_FROM][GR_Y], lv[GR_TO][GR_Y])
    y = lh[GR_FROM][GR_Y]
    f = min_y <= y and max_y >= y
    return f

def format_stone_pos(stone, axis = None):
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
            key = format_stone_pos(stone)
            p[key] = dict()
            p[key]['X'] = format_stone_pos(stone, GR_X)
            p[key]['Y'] = format_stone_pos(stone, GR_Y)
            p[key]['R'] = format_stone_pos(stone, GR_R)
            p[key]['A'] = format_stone_pos(stone, GR_A)
            p[key]['B'] = format_stone_pos(stone, GR_B)
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

# JGF to GR results
def jgf_to_gres(jgf):
    def sp(stones):
        p = np.zeros((len(stones), 5), dtype = np.int32)
        n = 0
        for key in stones.keys():
            stone = stones[key]
            p[n,GR_X] = int(stone['X'])
            p[n,GR_Y] = int(stone['Y'])
            p[n,GR_R] = int(stone['R'])
            p[n,GR_A] = ord(stone['A']) - ord('A') + 1
            p[n,GR_B] = int(stone['B'])
            n += 1
        return p

    res = dict()
    res[GR_BOARD_SIZE] = jgf['board_size']
    res[GR_EDGES] = [(0,0),(0,0)]
    res[GR_EDGES][0] = jgf['edges']['0']
    res[GR_EDGES][1] = jgf['edges']['1']
    res[GR_SPACING] = jgf['spacing']
    res[GR_STONES_B] = sp(jgf['black'])
    res[GR_STONES_W] = sp(jgf['white'])
    return res

# Resize the image proportionally so no side will exceed given max_size
# if f_upsize = False, images with size less than max_size are not upscaled
def resize(img, max_size, f_upsize = True, f_center = False, pad_color = (255, 255, 255)):
    """Resizes an image so neither of its sides will be bigger that max_size saving proportions

    Parameters:
        img         An OpenCv image
        max_size    Size to resize to
        f_upsize    If True, images less than max_size will be upsized
        f_center    If True, smaller images will be centered on bigger image with padding
        pad_color   Padding color

    Returns:
        Resized image
    """
    im, _, _ = resize3(img, max_size, f_upsize, f_center, pad_color)[0]
    return im

def resize2(img, max_size, f_upsize = True, f_center = False, pad_color = (255, 255, 255)):
    """Resizes an image so neither of its sides will be bigger that max_size saving proportions

    Parameters:
        img         An OpenCv image
        max_size    Size to resize to
        f_upsize    If True, images with size less than max_size will be upsized
        f_center    If True, smaller images will be centered on bigger image with padding
        pad_color   Padding color

    Returns:
        Resized image
        Scale [scale_x, scale_y]. Scale < 0 means image was downsized
    """
    im, scale, _ = resize3(img, max_size, f_upsize, f_center, pad_color)
    return im, scale

def resize3(img, max_size, f_upsize = True, f_center = False, pad_color = (255, 255, 255)):
    """Resizes an image so neither of its sides will be bigger that max_size saving proportions

    Parameters:
        img         An OpenCv image
        max_size    Size to resize to
        f_upsize    If True, images with size less than max_size will be upsized
        f_center    If True, smaller images will be centered on bigger image with padding
        pad_color   Padding color

    Returns:
        Resized image
        Scale [scale_x, scale_y]. Scale < 0 means image was downsized
        Offset [x, y]. If image was centered, offset of image location
    """
    im_size_max = np.max(img.shape[0:2])
    im_size_min = np.min(img.shape[0:2])
    im_scale = float(max_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    if not f_upsize and im_scale > 1.0:
       # Image size is less than max_size and upsize not specified
       if not f_center:
          # Nothing to do!
          return img, [1.0, 1.0], [0, 0]
       else:
          # Make a bigger image and center initial image on it
          if len(img.shape) > 2:
             im = np.full((max_size, max_size, img.shape[2]), pad_color, dtype=np.uint8)
          else:
             c = pad_color[0] if type(pad_color) is tuple else pad_color
             im = np.full((max_size, max_size), c, dtype=np.uint8)

          w = img.shape[CV_WIDTH]
          h = img.shape[CV_HEIGTH]
          dx = int((max_size - w)/2)
          dy = int((max_size - h)/2)

          im[dy:dy + h, dx:dx + w] = img

          return im, [1.0, 1.0], [dx, dy]
    else:
       # Perform normal resize
       im = cv2.resize(img, dsize = None, fx = im_scale, fy = im_scale)
       return im, [im_scale, im_scale], [0, 0]

# Calculate spacing
def board_spacing(edges, size):
    space_x = (edges[1][0] - edges[0][0]) / float(size-1)
    space_y = (edges[1][1] - edges[0][1]) / float(size-1)
    return space_x, space_y

