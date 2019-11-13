#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go board processing functions
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

import sys
from .grdef import *
from .utils import *
from .cv2_watershed import apply_watershed
from .py_accumulate import accumulate

import cv2
import numpy as np
import logging

# Find stones on a board
# Takes an image, recognition param dictionary, results dictionary and
# kind of stones been processed ('B' or 'W')
# Recognized stones are saved in a list in form of (X, Y, A, B, R),
# where (X,Y) are image coordinates, (A,B) - stone position, R - radius in pixels
# Several analysis parameters are also stored in the results dict
# The array is stored in the results dictionary (res)
def find_stones(src_img, params, res, f_bw):
    """Find stones on a board

       Parameters:
           img        An image to process
           params     Recognition parameters (see grdef.DEF_GR_PARAMS)
           res        Results dictionary (see grdef.GR_xxx)
           f_bw       Either B or W for black and white stones
       Returns:
            list of stones in form of (X, Y, A, B, R)
    """

    # Pre-filter: pyramid filtering
    def _apply_pmf(img, params, f_bw):
        n_pmf = params['PYRAMID_' + f_bw]
        if n_pmf == 0:
           logging.info("Filter skipped")
           return img
        else:
            pmf = cv2.pyrMeanShiftFiltering(img, 21, 51)
            res['IMG_PMF_' + f_bw] = pmf
            return pmf

    # Pre-filter: gray out
    def _apply_gray(img, params, f_bw):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pre-filter: extract channel
    def _apply_channel_mask(img, params, f_bw):
        b,g,r = cv2.split(img)
        if f_bw == 'B':
           return r
        else:
           return b

    # Pre-filter: thresholding
    def _apply_thresh(img, params, f_bw):
        n_thresh = params['STONES_THRESHOLD_' + f_bw]
        n_maxval = params['STONES_MAXVAL_' + f_bw]
        method = cv2.THRESH_BINARY
        if f_bw == 'W': method = cv2.THRESH_BINARY_INV

        ret, thresh = cv2.threshold(img, n_thresh, n_maxval, method)
        res['IMG_THRESH_' + f_bw] = thresh
        return thresh

    # Pre-filter: dilation
    def _apply_dilate(img, params, f_bw):
        n_iter = params['STONES_DILATE_' + f_bw]
        n_mask = params['HC_MASK_' + f_bw]
        if n_iter == 0:
           logging.info("Filter skipped")
           return img
        else:
           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
           kernel = cv2.resize(kernel, (n_mask,n_mask))
           return cv2.dilate(img, kernel,
                                  iterations=n_iter,
                                  borderType = cv2.BORDER_CONSTANT,
                                  borderValue = COLOR_BLACK)


    # Pre-filter: erode
    def _apply_erode(img, params, f_bw):
        n_iter = params['STONES_ERODE_' + f_bw]
        n_mask = params['HC_MASK_' + f_bw]
        if n_iter == 0:
           logging.info("Filter skipped")
           return img
        else:
           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
           kernel = cv2.resize(kernel, (n_mask,n_mask))
           return cv2.erode(img, kernel,
                                  iterations=n_iter,
                                  borderType = cv2.BORDER_CONSTANT,
                                  borderValue = COLOR_BLACK)

    # Pre-filter: blur
    def _apply_blur(img, params, f_bw):
        n_blur = params['BLUR_MASK_' + f_bw]
        if n_blur == 0:
           logging.info("Filter skipped")
           return img
        else:
            return cv2.blur(img, (n_blur, n_blur))

    # Pre-filter: luminosity equaliztion
    def _apply_clahe(img, params, f_bw):
        n_lum_eq = params['LUM_EQ']
        if n_lum_eq == 0:
            logging.info("Filter skipped")
            return img
        else:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Split channels
            l, a, b = cv2.split(lab)

            # Apply CLAHE to l_channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            # Merge back and convert to RGB color space
            merged = cv2.merge((cl,a,b))
            final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return final

    # Post-filter: houghCircle
    def _apply_houghc(img, filtered_img, params, f_bw, prev_stones):
        n_mindist = params['HC_MINDIST']
        n_maxrad = params['HC_MAXRADIUS']
        n_param2 = params['HC_SENSITIVITY_' + f_bw]
        return cv2.HoughCircles(filtered_img, cv2.HOUGH_GRADIENT,
                                       1,
                                       minDist = n_mindist,
                                       param1 = 100,
                                       param2 = n_param2,
                                       #minRadius = 3,
                                       maxRadius = n_maxrad)

    # Post-filter: watershed
    def _apply_watershed(img, filtered_img, params, f_bw, prev_stones):
        n_ws = params['WATERSHED_' + f_bw]

        if n_ws == 0 or prev_stones is None:
           logging.info("Filter skipped")
           return None
        else:
           #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           gray = _apply_channel_mask(img, params, f_bw)
           n_thresh = n_ws
           if n_thresh == 1: n_thresh = 190         # backward compatibility
           n_morph = params['WS_MORPH_' + f_bw]

           ws_stones, ws_img = apply_watershed(gray = gray, stones = prev_stones, \
                      n_thresh = n_thresh, f_bw = f_bw, n_morph = n_morph)

           res['IMG_WATERSHED_' + f_bw] = ws_img
           return ws_stones

    # Utility: combine stones from two arrays
    def _combine_stones(prev_stones, new_stones):
        if new_stones is None or len(new_stones) == 0:
           return prev_stones
        if prev_stones is None or len(prev_stones) == 0:
           return new_stones

        st_res = []
        min_r = sum([i[4] for i in new_stones]) / float(len(new_stones) * 2)
        for s1 in prev_stones:
            found = False
            for s2 in new_stones:
                if s1[GR_A] == s2[GR_A] and s1[GR_B] == s2[GR_B] and s2[GR_R] >= min_r:
                   st_res.append(s2)
                   found = True
                   break
            if not found:
                st_res.append(s1)

        return np.array(st_res)

    # Initialize filters
    def _init():
        return ({
            "PMF": _apply_pmf,
            'LUM_EQ': _apply_clahe,
            "CHANNEL": _apply_channel_mask,
            #"GRAY": _apply_gray,
            "THRESH": _apply_thresh,
            "STONES_DILATE": _apply_dilate,
            "STONES_ERODE": _apply_erode,
            "BLUR_MASK": _apply_blur
        },
        {
            "HOUGH_C": _apply_houghc,
            "WATERSHED": _apply_watershed
        })

    # Set up filters list
    (pre_filters, post_filters) = _init()

    # Process image with pre-filters
    filtered_img = src_img.copy()
    for f in pre_filters:
        logging.info("Applying pre-filter {} for color {}".format(f, f_bw))
        filtered_img = pre_filters[f](filtered_img, params, f_bw)
    res['IMG_MORPH_' + f_bw] = filtered_img

    # Process image with post-filters
    stones = None
    for f in post_filters:
        logging.info("Applying post-filter {} for color {}".format(f, f_bw))
        new_stones = post_filters[f](src_img, filtered_img, params, f_bw, stones)
        if new_stones is None:
           logging.info("No stones found by filter")
        else:
           if len(new_stones.shape) == 3: new_stones = new_stones[0]
           logging.info("Filter found {} stones".format(len(new_stones)))

           conv_stones = convert_xy(new_stones, res)
           stones = _combine_stones(stones, conv_stones)

    return stones

# Find board edges, spacing and size
def find_board(img, params, res):
    """Determine board parameters

       Parameters
           img         An image
           params      Recognition parameters (see grdef.DEF_GR_PARAMS)
           res         Results dictionary (see grdef.GR_xxx)
       Returns
           Board edges: tuple ([x1,y1], [x2,y2])
    """

    MIN_LINE_SPACE = 10

    def houghp_to_lines(lines):
        """ Transform HoughP results to lines array """
        ret = []
        if not lines is None:
            for i in lines:
                ret.append(((i[0][0],i[0][1]),(i[0][2],i[0][3])))
        return np.array(ret)

    def hough_to_lines(lines, shape):
        """ Transform Hough results to lines array"""
        ret = []
        if not lines is None:
            for i in lines:
                rho,theta = i[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho

                x1 = int(x0 + shape[CV_WIDTH]*(-b))
                y1 = int(y0 + shape[CV_HEIGTH]*(a))
                x2 = int(x0 - shape[CV_WIDTH]*(-b))
                y2 = int(y0 - shape[CV_HEIGTH]*(a))

                ret.append(((x1,y1),(x2,y2)))
        return np.array(ret)

    def unique_lines(a, delta = 10):
        """Return lines with are far from each other on more than a given distance"""
        if a is None: return None

        v = accumulate(a, lambda x,y: x if abs(y[0][0] - x[0][0]) < delta else y)
        l = [i for i in v]

        if l is None or len(l) == 0:
           return None
        else:
           return np.unique(np.array(l), axis = 0)

    def draw_rect(img, p1, p2):
        if p1[0] != p2[0] and p1[1] != p2[1]:
           cv2.rectangle(img, p1, p2, 0, -1)

    # Prepare gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res[GR_IMG_GRAY] = gray

    # Find edges
    n_minval = params['CANNY_MINVAL']
    n_maxval = params['CANNY_MAXVAL']
    n_apsize = params['CANNY_APERTURE']
    edges = cv2.Canny(gray, n_minval, n_maxval, apertureSize = n_apsize)
    res[GR_IMG_EDGES] = edges

    # Eliminate area defined by area mask. If mask is not provided, use default gap
##    m = None
##    if 'AREA_MASK' in params:
##       m = params['AREA_MASK']
##    if m is None or not type(m) is list:
##       d = int(MIN_EDGE_DIST/2)
##       m = [d, d, edges.shape[CV_WIDTH]-d, edges.shape[CV_HEIGTH]-d]
##
##    w = edges.shape[CV_WIDTH]
##    h = edges.shape[CV_HEIGTH]
##    draw_rect(edges, (0,0),       (m[0], h))
##    draw_rect(edges, (m[0],0),    (w, m[1]))
##    draw_rect(edges, (m[2],0),    (w, h))
##    draw_rect(edges, (m[0],m[3]), (m[2], h))

    # Run HoughLinesP, if its parameters are set
    # HoughLinesP detects line segments and may split a single line to multiple segments
    # The goal of running it is to remove small lines (less than minlen) which are,
    # for example, labels on board positions
    # If HoughLines is to run, its results will be used for further recognition as imput
    img_detect = edges
    n_rho = params['HL_RHO']
    n_theta = params['HL_THETA'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD']
    n_minlen = params['HL_MINLEN']
    if n_thresh > 0 and n_minlen > 0:
       lines = cv2.HoughLinesP(edges, n_rho, n_theta, n_thresh, minLineLength = n_minlen)
       lines = houghp_to_lines(lines)
       img_detect = make_lines_img(edges.shape, lines)
       res[GR_IMG_LINES] = img_detect
       img_detect = cv2.bitwise_not(img_detect)

    # Detect lines with HoughLines
    n_rho = params['HL_RHO2']
    n_theta = params['HL_THETA2'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD2']
    if n_thresh < 10: n_thresh = 90       #w/a for backward compt

    # HoughLines doesn't determine coordinates, but only direction (theta) and
    # distance from (0,0) point (rho)
    lines = cv2.HoughLines(img_detect, n_rho, n_theta, n_thresh)
    lines = sorted(lines, key = lambda f: f[0][0])

    # Find vertical/horizontal lines
    lines_v = [e for e in lines if e[0][1] == 0.0 if e[0][0] > 1]
    p = round(np.pi/2 * 100, 0)
    lines_h = [e for e in lines if round(e[0][1]*100,0) == p if e[0][0] > 1]

    # Remove duplicates (lines too close to each other)
    unique_v = unique_lines(lines_v)
    unique_h = unique_lines(lines_h)

    # Convert from (rho, theta) to coordinate-based lines
    lines_v = hough_to_lines(unique_v, img.shape)
    lines_h = hough_to_lines(unique_h, img.shape)
    vcross = len(lines_v)
    hcross = len(lines_h)
    res[GR_NUM_CROSS_H] = hcross
    res[GR_NUM_CROSS_W] = vcross

    # Detect edges
    if len(lines_h) == 0:
       logging.error("Cannot find horizontal lines, check params")
       return None
    if len(lines_v) == 0:
       logging.error("Cannot find vertical lines, check params")
       return None

    top_left = [int(lines_v[0][0][0]), int(lines_h[0][0][1])]
    bottom_right = [int(lines_v[-1][0][0]), int(lines_h[-1][0][1])]
    edges = (top_left, bottom_right)

    res[GR_EDGES] = edges
    logging.info("Edges within board area: {}".format(edges))

    # Draw a lines grid over gray image for debugging
    line_img = img1_to_img3(gray)
    line_img = make_lines_img(gray.shape, lines_v, width = 2, color = COLOR_RED, img = line_img)
    line_img = make_lines_img(gray.shape, lines_h, width = 2, color = COLOR_RED, img = line_img)
    res[GR_IMG_LINES2] = line_img

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
        # Take size which is more than minimum one (9)
        size = max(min(hcross, vcross),DEF_AVAIL_SIZES[0])
        if size > MAX_BOARD_SIZE:
            # Oops, take a default one
            logging.error("Cannot properly determine board size, fall back to default")
            size = DEF_BOARD_SIZE

    res[GR_BOARD_SIZE] = size
    logging.info("Detected board size: {}".format(size))

    # Calculate spacing
    space_x, space_y = board_spacing(edges, size)
    if space_x == 0 or space_y == 0:
       logging.error("Cannot determine spacing, check params")
       return None

    spacing = (space_x, space_y)
    res[GR_SPACING] = spacing
    logging.info("Detected spacing: {}".format(spacing))

    return edges

# Converts stone coordinates to stone positions
# Takes an array of coordinates created by find_stones and results dictionary
# Returns an array containg stones positions as well as board coordinates
# Board coordinates are stores as first two array items, board position - as 3,4
def convert_xy(coord, res):
    """Convert stone coordinates to board positions"""
    if coord is None or not GR_EDGES in res:
        return None

     # Make up an empty array
    stones = np.zeros((len(coord), 2), dtype = np.uint16)

    # Get edges and spacing
    edges = res[GR_EDGES]
    size = res[GR_BOARD_SIZE]
    space_x, space_y = res[GR_SPACING]

    # Loop through, converting board coordinates to integer positions
    # Radius is stored in dictiotary to retrieve later. This is kinda dumb way but I cannot find other one
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
        stones[i,GR_B] = size - stones_u[i, 1] + 1
        stones[i,GR_R] = old_coord[2]

    return stones

# Find a stone for given image coordinates
# Takes X and Y in image coordinates and a list of stones created by convert_xy
def find_coord(x, y, stones):
    """Returns index of a stone at given (X,Y) or None"""
    for i in stones:
        min_x = max(1, int(i[GR_X]) - int(i[GR_R]))
        min_y = max(1, int(i[GR_Y]) - int(i[GR_R]))
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
    """Main image processing function.

    Parameters:
        img    An image to process
        params Recognition parameters (see grdef.DEF_GR_PARAMS)

    Returns results dictionary (see grdef.GR_xxx)"""

    # Internal function: apply area mask
    def apply_area_mask(img, params):
        # Eliminate area defined by area mask. If mask is not provided, use default gap
        m = None
        if 'AREA_MASK' in params:
           m = params['AREA_MASK']
        if m is None or not type(m) is list:
           d = MIN_EDGE_DIST
           m = [d, d, img.shape[CV_WIDTH]-d, img.shape[CV_HEIGTH]-d]

        return get_image_area(img, m), [m[0], m[1]]

    # Internal function: duplicate elimination
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

    def offset_edges(edges, offset):
        if edges is None: return None
        edges[0][0] += offset[0]
        edges[0][1] += offset[1]
        edges[1][0] += offset[0]
        edges[1][1] += offset[1]
        return edges

    def offset_stones(stones, offset):
        if stones is None: return None
        for st in stones:
            st[GR_X] += offset[0]
            st[GR_Y] += offset[1]
        return stones

    res = dict()
    res[GR_IMAGE_SIZE] = img.shape[:2]
    logging.info("Image size is {}".format(res[GR_IMAGE_SIZE]))

    try:
        # Apply area mask
        img2, offset = apply_area_mask(img, params)

        # Find board edges, spacing, size
        board_edges = find_board(img2, params, res)
        if board_edges is None:
           logging.error('Edges not found, processing stopped')
           return None

        # Find stones
        black_stones = find_stones(img2, params, res, 'B')
        white_stones = find_stones(img2, params, res, 'W')

        # Elminate duplicates
        black_stones, white_stones = eliminate_duplicates(black_stones, white_stones)

        # Apply offset
        offset_edges(board_edges, offset)
        offset_stones(black_stones, offset)
        offset_stones(white_stones, offset)

    except:
        logging.exception("Processing error")
        raise

    # Store the results
    res[GR_STONES_B] = black_stones
    res[GR_STONES_W] = white_stones

    return res

# Creates a board image for given image shape and board size
# If recognition results are provided, plot them on the board
def generate_board(shape = DEF_IMG_SIZE, board_size = None, res = None, f_show_det = False):
    """Creates a board image for given image shape and board size
       If recognition results are provided, plot them on the board
       if f_show_det is True, also plots detections
       Returns generated image"""

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
                x1 = int(edges[GR_FROM][GR_X] + (i[GR_A]-1) * space_x)
                y1 = int(edges[GR_FROM][GR_Y] + (board_size - i[GR_B]) * space_y)
                cv2.circle(img, (x1,y1), r, COLOR_BLACK, -1)

                if f_show_det:
                   x2 = i[GR_X]
                   y2 = i[GR_Y]
                   r2 = i[GR_R]
                   cv2.circle(img, (x2,y2), r2, (0,0,255), 1)

        if white_stones is not None:
            for i in white_stones:
                x1 = int(edges[GR_FROM][GR_X] + (i[GR_A]-1) * space_x)
                y1 = int(edges[GR_FROM][GR_Y] + (board_size - i[GR_B]) * space_y)
                cv2.circle(img, (x1,y1), r, COLOR_BLACK, 1)
                cv2.circle(img, (x1,y1), r-1, COLOR_WHITE, -1)

                if f_show_det:
                   x2 = i[GR_X]
                   y2 = i[GR_Y]
                   r2 = i[GR_R]
                   cv2.circle(img, (x2,y2), r2, (0,0,255), 1)

    return img


