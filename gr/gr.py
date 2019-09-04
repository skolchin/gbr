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
    from cv2_watershed import apply_watershed
else:
    from gr.grdef import *
    from gr.utils import *
    from gr.cv2_watershed import apply_watershed
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
    params: recognition parameters (see grdef.DEF_GR_PARAMS)
    res: results dictionary (see grdef.GR_xxx)
    f_bw: either B or W for black and white stones"""

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
        logging.info("Applying pre-filter {} for {} color".format(f, f_bw))
        filtered_img = pre_filters[f](filtered_img, params, f_bw)
    res['IMG_MORPH_' + f_bw] = filtered_img

    # Process image with post-filters
    stones = None
    for f in post_filters:
        logging.info("Applying post-filter {} for {} color".format(f, f_bw))
        new_stones = post_filters[f](src_img, filtered_img, params, f_bw, stones)
        if new_stones is None:
           logging.info("No stones found after applying post-filter")
           break;
        else:
           if len(new_stones.shape) == 3: new_stones = new_stones[0]
           logging.info("Stones found: {}".format(len(new_stones)))

           conv_stones = convert_xy(new_stones, res)
           stones = _combine_stones(stones, conv_stones)

    return stones

# Find board edges, spacing and size
# Takes an image, recognition param dictionary and results dictionary
# Finds:
#   board edges: a tuple ((xmin,ymin),(xmax,ymax)) in image coords
#   board size: single value
#   board net spacing: a tuple (sx, sy) in image coordinates
# Some other analysis parameters are also stored in the results dict
# Returns board edges
def find_board(img, params, res):
    """Determine board parameters
    params: recognition parameters (see grdef.DEF_GR_PARAMS)
    res: results dictionary (see grdef.GR_xxx)"""

    MIN_LINE_LEN = 10
    MIN_EDGE_DIST = 6

    def is_horizontal(line):
        """ Check whether the line is horizontal """
        return abs(line[GR_FROM][GR_Y] - line[GR_TO][GR_Y]) < 3 \
                    and abs(line[GR_FROM][GR_X] - line[GR_TO][GR_X]) > MIN_LINE_LEN

    def is_vertical(line):
        """ Check whether the line is vertical """
        return abs(line[GR_FROM][GR_X] - line[GR_TO][GR_X]) < 3 \
                    and abs(line[GR_FROM][GR_Y] - line[GR_TO][GR_Y]) > MIN_LINE_LEN

    def cmp_less(pt, line, axis):
        """ Returns a line with min origin amoung the two """
        if pt is None:
            return line[GR_FROM]
        elif line[GR_FROM][axis] < pt[axis]:
            return line[GR_FROM]
        else:
            return pt

    def cmp_greater(pt, line, axis):
        """ Returns a line with max ending amoung the two """
        if pt is None:
            return line[GR_TO]
        elif line[GR_TO][axis] > pt[axis]:
            return line[GR_TO]
        else:
            return pt

    def find_line(lines, axis, orient_fun, cmp_fun):
        """ Find a line of given orientation and matching functions """
        found = None
        for line in lines:
            if orient_fun(line):
                found = cmp_fun(found, line, axis)
        return found

    def houghp_to_lines(lines):
        """ Transform HoughP results to lines array """
        ret = []
        if lines is None:
           return ret
        for i in lines:
            ret.append(((i[0][0],i[0][1]),(i[0][2],i[0][3])))
        return ret

    def hough_to_lines(lines, shape):
        """ Transform Hough results to lines array """
        ret = []
        if lines is None:
           return ret
        for i in lines:
            rho,theta = i[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho

            x1 = int(x0 + shape[0]*(-b))
            y1 = int(y0 + shape[1]*(a))
            x2 = int(x0 - shape[0]*(-b))
            y2 = int(y0 - shape[1]*(a))

            ret.append(((x1,y1),(x2,y2)))
        return ret

    # Main function starts here

    # Graying out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res[GR_IMG_GRAY] = gray

    # Find edges
    n_minval = params['CANNY_MINVAL']
    n_maxval = params['CANNY_MAXVAL']
    n_apsize = params['CANNY_APERTURE']
    edges = cv2.Canny(gray, n_minval, n_maxval, apertureSize = n_apsize)

    # Eliminate area close to edges in order not to give false line detections
    dx = int(MIN_EDGE_DIST/2)
    edges = cv2.rectangle(edges, (dx,dx),
                (edges.shape[CV_WIDTH]-dx, edges.shape[CV_HEIGTH]-dx), COLOR_BLACK, MIN_EDGE_DIST)

    # Detect lines with HoughLinesP
    # This function returns line segments as (N,4) array
    # It might split a line to several segments thus not allowing to find its
    # true origin or ending
    n_rho = params['HL_RHO']
    n_theta = params['HL_THETA'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD']
    lines = cv2.HoughLinesP(edges, n_rho, n_theta, n_thresh)
    lines = houghp_to_lines(lines)

    nlin = len(lines)
    res[GR_NUM_LINES] = nlin
    logging.info("Number of lines after HoughLinesP: {}".format(nlin))

    # Find first and last horizontal and vertical line
    hl = (find_line(lines, GR_Y, is_horizontal, cmp_less),
            find_line(lines, GR_Y, is_horizontal, cmp_greater))
    vl = (find_line(lines, GR_X, is_vertical, cmp_less),
            find_line(lines, GR_X, is_vertical, cmp_greater))

    # Determine edges
    if hl[0] is None:
        corner1 = (int(vl[0][0]), int(vl[1][1]))
        corner2 = (int(vl[0][1]), int(vl[1][0]))
    else:
        corner1 = (int(min(hl[0][0], vl[0][0])), int(min(hl[0][1], vl[0][1])))
        corner2 = (int(max(hl[1][0], vl[1][0])), int(max(hl[1][1], vl[1][1])))

    brd_edges = (corner1, corner2)
    res[GR_EDGES] = brd_edges
    logging.info("Edges detected: {}".format(brd_edges))

##    for i in lines:
##        x1 = i[GR_FROM][GR_X]
##        y1 = i[GR_FROM][GR_Y]
##        x2 = i[GR_TO][GR_X]
##        y2 = i[GR_TO][GR_Y]
##
##        if (abs(x1 - x2) < 3 and abs(y1 - y2) > 10) or (abs(y1 - y2) < 3 and abs(x1 - x2) > 10):
##            # Vertical or horizontal line
##            if (x1 < xmin or xmin == -1):
##                xmin = x1
##            if (y1 < ymin or ymin == -1):
##                ymin = y1
##            if (x2 > xmax or xmax == -1):
##                xmax = x2
##            if (y2 > ymax or ymax == -1):
##                ymax = y2
##
##    brd_edges = ((int(xmin), int(ymin)), (int(xmax), int(ymax)))

    # Redraw the image to contain only lines detected by HoughLineP
    lines_img = make_lines_img(edges.shape, lines)
    res[GR_IMG_LINES] = lines_img

    # Detecting board size using HoughLine
    n_rho = params['HL_RHO2']
    n_theta = params['HL_THETA2'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD2']

    lines_img2 = cv2.bitwise_not(lines_img)
    lines2 = cv2.HoughLines(lines_img2, n_rho, n_theta, n_thresh)
    lines2 = hough_to_lines(lines2, lines_img2.shape)
    logging.info("Number of lines after HoughLines: {}".format(len(lines2)))

    # Collect horizontal/vertical lines
    hlin = []
    vlin = []
    for i in lines2:
        if is_vertical(i): vlin.append(i)
        if is_horizontal(i): hlin.append(i)

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
        # Take size which is more than minimum one (9)
        size = max(min(hcross, vcross),DEF_AVAIL_SIZES[0])
        if size > MAX_BOARD_SIZE:
            # Oops, take a default one
            logging.error("Cannot properly determine board size, fall back to default")
            size = DEF_BOARD_SIZE

    res[GR_BOARD_SIZE] = size
    logging.info("Detected board size: {}".format(size))

     # Make a debug image
    lines_img2 = make_lines_img(gray.shape, hpos)
    lines_img2 = make_lines_img(gray.shape, vpos, img = lines_img2)
    res[GR_IMG_LINES2] = lines_img2

    # Calculate spacing
    space_x, space_y = board_spacing(brd_edges, size)
    spacing = (space_x, space_y)
    res[GR_SPACING] = spacing
    logging.info("Detected spacing: {}".format(spacing))

    return brd_edges

# Converts stone coordinates to stone positions
# Takes an array of coordinates created by find_stones and results dictionary
# Returns an array containg stones positions as well as board coordinates
# Board coordinates are stores as first two array items, board position - as 3,4
def convert_xy(coord, res):
    """Convert stone coordinates to board positions"""
    if coord is None:
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
        stones[i,GR_B] = stones_u[i, 1]
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
    """Main image processing function
    img: image to process
    params: recognition parameters (see grdef.DEF_GR_PARAMS)
    returns results dictionary (see grdef.GR_xxx)"""

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

    res = dict()
    res[GR_IMAGE_SIZE] = img.shape[:2]
    logging.info("Image size is {}".format(res[GR_IMAGE_SIZE]))

    try:
        # Find board edges, spacing, size
        board_edges = find_board(img, params, res)

        # Find stones
        black_stones = find_stones(img, params, res, 'B')
        white_stones = find_stones(img, params, res, 'W')

        # Elminate duplicates
        black_stones, white_stones = eliminate_duplicates(black_stones, white_stones)
    except:
        logging.error(sys.exc_info()[1])
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


