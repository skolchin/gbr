#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go board processing functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import sys
if sys.version_info[0] < 3:
    from grdef import *
    from utils import *
else:
    from gr.grdef import *
    from gr.utils import *
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

# Find stones on a board
# Takes an image, recognition param dictionary, results dictionary and
# kind of stones been processed ('B' or 'W')
# Recognized stones are saved in a list in form of (X, Y, A, B, R),
# where (X,Y) are image coordinates, (A,B) - stone position, R - radius in pixels
# Several analysis parameters are also stored in the results dict
# The array is stored in the results dictionary and returned
def find_stones(img, params, res, f_bw):

    # Apply watershed transformation
    # The algorithm was originally developed by Adrian Rosebrock (PyImageSearch.com)
    # and adopted so it currently uses stone coordinations as an indicators of peaks
    # instead of original "max peak value" method
    # TODO: rewrite to use only OpenCV methods
    def apply_watershed(img, stones, n_thresh, f_bw):
        WS_MIN = 190
        WS_MAX = 255

        if f_bw == 'B':
           # To have watershed properly determine black stones, board dividers
           # have to be removed with dilation, source image converted to negative
           kernel  = np.ones((3,3),np.uint8)
           img2 = cv2.dilate(img,kernel,iterations=1)
           #img2 = cv2.erode(img2,kernel,iterations=1)
           img2 = cv2.bitwise_not(img2)
           ret, t2 = cv2.threshold(img2, n_thresh, WS_MAX, cv2.THRESH_BINARY)
        else:
           # White stones, normal thresholding
           ret, t2 = cv2.threshold(img, n_thresh, WS_MAX, cv2.THRESH_BINARY)

        # Apply distance transformation
        D = ndimage.distance_transform_edt(t2)

        # Use stones coordinations as peaks
        peaks = np.zeros(t2.shape[:2], dtype = np.bool)
        for st in stones:
            x = int(st[0])
            y = int(st[1])
            peaks[y,x] = True
            if img[y,x] < n_thresh:
               # Central point is not white
               # Move little bit around to find proper point
               for xt in range(7):
                   x1 = x + xt - 3
                   f = False
                   for yt in range(7):
                       y1 = y + yt - 3
                       if img[y1,x1] >= n_thresh:
                          peaks[y,x] = False
                          peaks[y1,x1] = True
                          f = True
                          break
                   if f: break

        # Set the markers and apply watershed
        markers = ndimage.label(peaks, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=t2)

        # Collect results
        rt = []
        for label in np.unique(labels):
            if label == 0: continue

            mask = np.zeros(t2.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            rt.append ([int(x), int(y), int(r)])

        return np.array(rt)

    # Params
    n_thresh = params['STONES_THRESHOLD_' + f_bw]
    n_maxval = params['STONES_MAXVAL_' + f_bw]
    n_iter_d = params['STONES_DILATE_' + f_bw]
    n_iter_e = params['STONES_ERODE_' + f_bw]
    n_mindist = params['HC_MINDIST']
    n_maxrad = params['HC_MAXRADIUS']
    n_param2 = params['HC_SENSITIVITY_' + f_bw]
    n_mask = params['HC_MASK_' + f_bw]
    n_blur = params['BLUR_MASK_' + f_bw]
    n_watershed = params['WATERSHED_' + f_bw]

    # Create thresholded image
    thresh = None
    if (f_bw == 'B'):
        ret, thresh = cv2.threshold(img, n_thresh, n_maxval, cv2.THRESH_BINARY)
    else:
        ret, thresh = cv2.threshold(img, n_thresh, n_maxval, cv2.THRESH_BINARY_INV)

    res['IMG_THRESH_' + f_bw] = thresh

    # Dilate and erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    kernel = cv2.resize(kernel, (n_mask,n_mask))

    stones_img = thresh.copy()
    if n_iter_d > 0:
        stones_img = cv2.dilate(stones_img, kernel,
                                            iterations=n_iter_d,
                                            borderType = cv2.BORDER_CONSTANT,
                                            borderValue = COLOR_BLACK)
    if n_iter_e > 0:
        stones_img = cv2.erode(stones_img, kernel,
                                           iterations=n_iter_e,
                                           borderType = cv2.BORDER_CONSTANT,
                                           borderValue = COLOR_BLACK)

    # Add some blur and sharpen the image to smooth the edges
    if n_blur > 0:
        stones_img = cv2.blur(stones_img, (n_blur, n_blur))

    res['IMG_MORPH_' + f_bw] = stones_img

    # Find stones
    stones = cv2.HoughCircles(stones_img, cv2.HOUGH_GRADIENT,
                                          1,
                                          minDist = n_mindist,
                                          param1 = 100,
                                          param2 = n_param2,
                                          #minRadius = 3,
                                          maxRadius = n_maxrad)

    if stones is None:
        return None
    else:
        # Determine board positions
        stones = convert_xy(stones[0], res)
        if n_watershed == 0:
            return stones
        else:
            # Apply watershed
            ws_stones = apply_watershed(img, stones, n_thresh, f_bw)
            ws_stones = convert_xy(ws_stones, res)

            # Combine stones from both sources
            st_res = []
            min_r = sum([i[4] for i in ws_stones]) / float(len(ws_stones) * 2)
            for s1 in stones:
                found = False
                for s2 in ws_stones:
                    # Use watershed's results only if determined radius is not too small
                    if s1[GR_A] == s2[GR_A] and s1[GR_B] == s2[GR_B] and s2[GR_R] >= min_r:
                       st_res.append(s2)
                       found = True
                       break
                if not found:
                    st_res.append(s1)

            return np.array(st_res)

# Find board edges, spacing and size
# Takes an image, recognition param dictionary and results dictionary
# Finds:
#   board edges: a tuple ((xmin,ymin),(xmax,ymax)) in image coords
#   board size: single value
#   board net spacing: a tuple (sx, sy) in image coordinates
# Some other analysis parameters are also stored in the results dict
# Returns board edges
def find_board(img, params, res):
    # Find edges
    n_minval = params['CANNY_MINVAL']
    n_maxval = params['CANNY_MAXVAL']
    n_apsize = params['CANNY_APERTURE']
    edges = cv2.Canny(img, n_minval, n_maxval, apertureSize = n_apsize)

    # Find lines, 1st pass
    n_rho = params['HL_RHO']
    n_theta = params['HL_THETA'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD']

    #n_iter = params['HL_DILATE']
    #if n_iter > 0:
    #   kernel = np.ones((3,3),np.uint8)
    #   edges = cv2.dilate(edges, kernel, iterations=n_iter)

    lines = cv2.HoughLinesP(edges, n_rho, n_theta, n_thresh)
    lines = houghp_to_lines(lines)

    # Remove lines too close to board edges
    lines = clear_lines(edges.shape, lines)
    lines_img = make_lines_img(edges.shape, lines)
    nlin = len(lines)

    res[GR_NUM_LINES] = nlin
    res[GR_IMG_LINES] = lines_img

    # Find min/max coordinates - which are edges
    xmin = -1
    xmax = -1
    ymin = -1
    ymax = -1

    for i in lines:
        x1 = i[GR_FROM][GR_X]
        y1 = i[GR_FROM][GR_Y]
        x2 = i[GR_TO][GR_X]
        y2 = i[GR_TO][GR_Y]

        if (abs(x1 - x2) < 3 or abs(y1 - y2) < 3):
            # Vertical or horizontal line
            if (x1 < xmin or xmin == -1):
                xmin = x1
            if (y1 < ymin or ymin == -1):
                ymin = y1
            if (x2 > xmax or xmax == -1):
                xmax = x2
            if (y2 > ymax or ymax == -1):
                ymax = y2

    brd_edges = ((int(xmin), int(ymin)), (int(xmax), int(ymax)))
    res[GR_EDGES] = brd_edges

    # Detecting board size
    n_rho = params['HL_RHO2']
    n_theta = params['HL_THETA2'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD2']

    lines_img2 = cv2.bitwise_not(lines_img)
    lines2 = cv2.HoughLines(lines_img2, n_rho, n_theta, n_thresh)
    lines2 = hough_to_lines(lines2, lines_img2.shape)

    hlin = []
    vlin = []

    for i in lines2:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]

        # Collect all horizontal and vertical lines
        if abs(x1 - x2) < 3 and y1 != y2:
            # vertical
            vlin.append(i)
        elif abs(y1 - y2) < 3 and x1 != x2:
            # horizontal
            hlin.append(i)

    # Get unique vertical line positions
    vpos = remove_nearest(vlin, axis1 = 0, axis2 = 0)
    vcross = len(vpos)
    hpos = remove_nearest(hlin, axis1 = 0, axis2 = 1)
    hcross = len(hpos)
    res[GR_NUM_CROSS_H] = hcross
    res[GR_NUM_CROSS_W] = vcross

    # Determine board size
    # First check both sizes are not more or less than 1 point from any of predefined sizes
    size = None
    for n in DEF_AVAIL_SIZES:
        if abs(hcross-n) < 2 and abs(vcross-n) < 2:
            size = n
            break

    if size is None:
        # Repeat but now check only one side
        for n in DEF_AVAIL_SIZES:
            if abs(hcross-n) < 2 or abs(vcross-n) < 2:
                size = n
                break

    if size is None:
        # Take size which is more than minimum one
        if hcross > DEF_AVAIL_SIZES[0] and vcross > DEF_AVAIL_SIZES[0]:
            size = min(hcross, vcross)
        elif hcross > DEF_AVAIL_SIZES[0]:
            size = hcross
        elif vcross > DEF_AVAIL_SIZES[0]:
            size = vcross

    if size is None:
        # Oops, take a default one
        print("Cannot properly determine board size, fall back to default one")
        size = DEF_BOARD_SIZE

    res[GR_BOARD_SIZE] = size

     # Make a debug image
    lines_img2 = make_lines_img(img.shape, hpos)
    lines_img2 = make_lines_img(img.shape, vpos, img = lines_img2)
    res[GR_IMG_LINES2] = lines_img2

    # Calculate spacing
    space_x, space_y = board_spacing(brd_edges, size)
    res[GR_SPACING] = (space_x, space_y)

    print("Edges:({},{}), ({},{}), crosses: {}, {}, size: {}, spaces: ({}, {})".format(
                          brd_edges[0][0], brd_edges[0][1],
                          brd_edges[1][0], brd_edges[1][0],
                          hcross, vcross,
                          size,
                          round(space_x,2), round(space_y,2)))
    return brd_edges

# Converts stone coordinates to stone positions
# Takes an array of coordinates created by find_stones and results dictionary
# Returns an array containg stones positions as well as board coordinates
# Board coordinates are stores as first two array items, board position - as 3,4
def convert_xy(coord, res):
    if (coord is None):
        return np.empty([0,0,0,0])
    else:
         # Make up an empty array
        stones = np.zeros((len(coord), 2), dtype = np.uint16)

        # Get edges and spacing
        edges = res[GR_EDGES]
        size = res[GR_BOARD_SIZE]
        space_x, space_y = res[GR_SPACING]

        # Loop through, converting board coordinates to integer positions
        # Radius is stored in dictiotary to retrieve later
        # This is kinda dumb way but I cannot find other one
        rd = dict()
        for i in range(len(coord)):
            x = coord[i,0] - edges[0][0]
            y = coord[i,1] - edges[0][1]

            stones[i,0] = int(round(x / space_x, 0)) + 1
            stones[i,1] = int(round(y / space_y, 0)) + 1
            rd[str(stones[i,0]) + "_" + str(stones[i,1])] = coord[i]

        # Remove duplicates
        stones_u = unique_rows(stones)

        # Calculate coordinates for stones left in the list
        stones = np.zeros((len(stones_u), 5), dtype = np.uint16)

        for i in range(len(stones_u)):
            old_coord = rd[str(stones_u[i,0]) + "_" + str(stones_u[i,1])]
            stones[i,GR_X] = old_coord[0]
            stones[i,GR_Y] = old_coord[1]
            stones[i,GR_A] = stones_u[i, 0]
            stones[i,GR_B] = stones_u[i, 1]
            stones[i,GR_R] = old_coord[2]

        return stones

# Find a stone for given image coordinates
# Takes X and Y in image coordinates and a list of stones created by convert_xy
def find_coord(x, y, coord):
    for i in coord:
        min_x = int(i[GR_X]) - int(i[GR_R])
        if min_x < 1: min_x = 1
        min_y = int(i[GR_Y]) - int(i[GR_R])
        if min_y < 1: min_y = 1
        max_x = i[GR_X] + i[GR_R]
        max_y = i[GR_Y] + i[GR_R]
        if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
            return i

    return None


# Board image processing main function
# Takes board image and recognition params
# Returns a dictionary containing recognition results
# Dict keys are defined in grdef module
def process_img(img, params):
    res = dict()
    res[GR_IMAGE_SIZE] = img.shape[:2]

    # Graying out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res[GR_IMG_GRAY] = gray
    b,g,r = cv2.split(img)
    res[GR_IMG_BLUE] = b
    res[GR_IMG_RED] = r

    # Find board edges, spacing, size
    board_edges = find_board(gray, params, res)

    # Find stones
    black_stones = find_stones(r, params, res, 'B')
    white_stones = find_stones(b, params, res, 'W')

    # Elminate duplicates
    black_stones, white_stones = eliminate_duplicates(black_stones, white_stones)

    # Store the results
    res[GR_STONES_B] = black_stones
    res[GR_STONES_W] = white_stones

    return res

# Creates a board image for given image shape and board size
# If recognition results are provided, plot them on the board
def generate_board(shape = DEF_IMG_SIZE, board_size = None, res = None, f_show_det = False):

    # Prepare params
    if board_size is None:
        if res is None:
            board_size = DEF_BOARD_SIZE
        else:
            board_size = res[GR_BOARD_SIZE]

    if res is None:
        edges = ((14,14),(shape[CV_WIDTH]-14, shape[CV_HEIGTH]-14))
        space_x, space_y = board_spacing(edges, board_size)
    else:
        edges = res[GR_EDGES]
        space_x = res[GR_SPACING][GR_X]
        space_y = res[GR_SPACING][GR_Y]

    # Make up empty image
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:] = DEF_IMG_COLOR
    #print("Image shape: {}".format(img.shape))

    # Draw the lines
    #print("Vertical")
    for i in range(board_size):
        x1 = int(edges[GR_FROM][GR_X] + (i * space_x))
        y1 = int(edges[GR_FROM][GR_Y])
        x2 = x1
        y2 = int(edges[GR_TO][GR_Y])
        cv2.line(img,(x1,y1),(x2,y2),COLOR_BLACK,1)

    #print("Horizontal")
    for i in range(board_size):
        x1 = int(edges[GR_FROM][GR_X])
        y1 = int(edges[GR_FROM][GR_Y] + (i * space_y))
        x2 = int(edges[GR_TO][GR_X])
        y2 = y1
        cv2.line(img, (x1,y1), (x2,y2), COLOR_BLACK, 1)

    # Draw the stones
    if res is not None:
        black_stones = res.get(GR_STONES_B)
        white_stones = res.get(GR_STONES_W)
        r = max(int(min(space_x, space_y) / 2) - 1, 5)

        if black_stones is not None:
            for i in black_stones:
                x1 = int(edges[GR_FROM][GR_X] + (i[GR_A]-1) * space_x)      #int(i[GR_X])
                y1 = int(edges[GR_FROM][GR_Y] + (i[GR_B]-1) * space_y)      #int(i[GR_Y])
                cv2.circle(img, (x1,y1), r, COLOR_BLACK, -1)

                if f_show_det:
                   x2 = i[GR_X]
                   y2 = i[GR_Y]
                   r2 = i[GR_R]
                   cv2.circle(img, (x2,y2), r2, (0,0,255), 1)

        if white_stones is not None:
            for i in white_stones:
                x1 = int(edges[GR_FROM][GR_X] + (i[GR_A]-1) * space_x)      #int(i[GR_X])
                y1 = int(edges[GR_FROM][GR_Y] + (i[GR_B]-1) * space_y)      #int(i[GR_Y])
                cv2.circle(img, (x1,y1), r, COLOR_BLACK, 1)
                cv2.circle(img, (x1,y1), r-1, COLOR_WHITE, -1)

                if f_show_det:
                   x2 = i[GR_X]
                   y2 = i[GR_Y]
                   r2 = i[GR_R]
                   cv2.circle(img, (x2,y2), r2, (0,0,255), 1)

    return img

def eliminate_duplicates(bs, ws):
    if ws is None or bs is None:
        return bs, ws

    # Priority for white stones
    for st in ws:
        px = st[GR_A]
        py = st[GR_B]
        for i in range(len(bs)):
            if px == bs[i,GR_A] and py == bs[i, GR_B]:
                bs = np.delete(bs, i, axis = 0)
                break;
    return bs, ws

