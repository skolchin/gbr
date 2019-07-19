#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go board processing functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import grdef
import grutils
import cv2
import numpy as np

# Find stones on a board
# Takes an image, recognition param dictionary, results dictionary and
# kind of stones been processed ('B' or 'W')
# Recognized stones are saved in a list in form of (X, Y, R),
# where X,Y are image coordinates, R - radius in pixels
# Several analysis parameters are also stored in the results dict
# The array is stored in the results dictionary and returned
def find_stones(img, params, res, f_bw):

    # Make a thresholded image
    n_thresh = params['STONES_THRESHOLD_' + f_bw]
    n_maxval = params['STONES_MAXVAL_' + f_bw]
    n_iter_d = params['STONES_DILATE_' + f_bw]
    n_iter_e = params['STONES_ERODE_' + f_bw]
    n_mindist = params['HC_MINDIST']
    n_maxrad = params['HC_MAXRADIUS']
    n_param2 = params['HC_SENSITIVITY_' + f_bw]
    n_mask = params['HC_MASK_' + f_bw]
    n_blur = params['BLUR_MASK_' + f_bw]

    thresh = None
    if (f_bw == 'B'):
       ret, thresh = cv2.threshold(img, n_thresh, n_maxval, cv2.THRESH_BINARY)
    else:
       ret, thresh = cv2.threshold(img, n_thresh, n_maxval, cv2.THRESH_BINARY_INV)

    res['IMG_THRESH_' + f_bw] = thresh

    # Dilate and erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
    kernel = cv2.resize(kernel, (n_mask, n_mask))

    stones_img = thresh
    if n_iter_d > 0:
       stones_img = cv2.dilate(stones_img, kernel,
                                           iterations=n_iter_d,
                                           borderType = cv2.BORDER_CONSTANT,
                                           borderValue = grdef.COLOR_BLACK)
    if n_iter_e > 0:
       stones_img = cv2.erode(stones_img, kernel,
                                          iterations=n_iter_e,
                                          borderType = cv2.BORDER_CONSTANT,
                                          borderValue = grdef.COLOR_BLACK)

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

    if (stones is None):
       return None
    else:
         return stones[0]

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
    lines = grutils.houghp_to_lines(lines)

    # Remove lines too close to board edges
    lines = grutils.clear_lines(edges.shape, lines)
    lines_img = grutils.make_lines_img(edges.shape, lines)
    nlin = len(lines)

    res[grdef.GR_NUM_LINES] = nlin
    res[grdef.GR_IMG_LINES] = lines_img
    print ("Lines found: " + str(nlin))

    # Find min/max coordinates - which are edges
    xmin = -1
    xmax = -1
    ymin = -1
    ymax = -1

    for i in lines:
        x1 = i[grdef.GR_FROM][grdef.GR_X]
        y1 = i[grdef.GR_FROM][grdef.GR_Y]
        x2 = i[grdef.GR_TO][grdef.GR_X]
        y2 = i[grdef.GR_TO][grdef.GR_Y]

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

    brd_edges = ((xmin, ymin), (xmax, ymax))
    res[grdef.GR_EDGES] = brd_edges

    # Detecting board size
    n_rho = params['HL_RHO2']
    n_theta = params['HL_THETA2'] * np.pi / 180
    n_thresh = params['HL_THRESHOLD2']

    lines_img2 = cv2.bitwise_not(lines_img)
    lines2 = cv2.HoughLines(lines_img2, n_rho, n_theta, n_thresh)
    lines2 = grutils.hough_to_lines(lines2, lines_img2.shape)

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
    vpos = grutils.remove_nearest(vlin, axis1 = 0, axis2 = 0)
    vcross = len(vpos)
    hpos = grutils.remove_nearest(hlin, axis1 = 0, axis2 = 1)
    hcross = len(hpos)
    res[grdef.GR_NUM_CROSS_H] = hcross
    res[grdef.GR_NUM_CROSS_W] = vcross

    # Determine board size
    # First check both sizes are not more or less than 1 point from any of predefined sizes
    size = None
    for n in grdef.DEF_AVAIL_SIZES:
        if abs(hcross-n) < 2 and abs(vcross-n) < 2:
           size = n
           break

    if size is None:
       # Repeat but now check only one side
        for n in grdef.DEF_AVAIL_SIZES:
           if abs(hcross-n) < 2 or abs(vcross-n) < 2:
              size = n
              break

    if size is None:
       # Take size which is more than minimum one
       if hcross > grdef.DEF_AVAIL_SIZES[0] and vcross > grdef.DEF_AVAIL_SIZES[0]:
          size = min(hcross, vcross)
       elif hcross > grdef.DEF_AVAIL_SIZES[0]:
          size = hcross
       elif vcross > grdef.DEF_AVAIL_SIZES[0]:
          size = vcross

    if size is None:
       # Oops, take a default one
       print("Cannot properly determine board size, fall back to default one")
       size = grdef.DEF_BOARD_SIZE

    res[grdef.GR_BOARD_SIZE] = size

    # Make a debug image
    lines_img2 = grutils.make_lines_img(img.shape, hpos)
    lines_img2 = grutils.make_lines_img(img.shape, vpos, img = lines_img2)
    res[grdef.GR_IMG_LINES2] = lines_img2

    # Calculate spacing
    space_x = (brd_edges[1][0] - brd_edges[0][0]) / (size - 1)
    space_y = (brd_edges[1][1] - brd_edges[0][1]) / (size - 1)
    res[grdef.GR_SPACING] = (space_x, space_y)

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
        edges = res[grdef.GR_EDGES]
        size = res[grdef.GR_BOARD_SIZE]
        space_x, space_y = res[grdef.GR_SPACING]

        # Loop through, converting board coordinates to integer positions
        for i in range(len(coord)):
            x = coord[i,0] - edges[0][0]
            y = coord[i,1] - edges[0][1]

            stones[i,0] = int(round(x / space_x, 0)) + 1
            stones[i,1] = int(round(y / space_y, 0)) + 1

            print("{}: ({}, {}) -> {}, {}".format(i,
                       coord[i,0], coord[i,1], stones[i,0], stones[i,1]))

        # Remove duplicates
        stones_u = grutils.unique_rows(stones)

        # Calculate coordinates for stones left in the list
        stones = np.zeros((len(stones_u), 4), dtype = np.uint16)
        for i in range(len(stones_u)):
            stones[i,0] = (stones_u[i, 0]-1) * space_x + edges[0][0]
            stones[i,1] = (stones_u[i, 1]-1) * space_y + edges[0][1]
            stones[i,2] = stones_u[i, 0]
            stones[i,3] = stones_u[i, 1]

        return stones

# Find a stone for given image coordinates
# Takes X and Y in image coordinates and a list of stones created by convert_xy
def find_coord(x, y, coord):
    for i in coord:
        min_x = i[0] - 7
        min_y = i[1] - 7
        max_x = i[0] + 7
        max_y = i[1] + 7
        if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
           return i

    return (-1, -1, -1, -1)


# Board image processing main function
# Takes board image and recognition params
# Returns a dictionary containing recognition results
# Dict keys are defined in grdef module
def process_img(img, params):
    res = dict()

    # Graying out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res[grdef.GR_IMG_GRAY] = gray
    b,g,r = cv2.split(img)
    res[grdef.GR_IMG_BLUE] = b
    res[grdef.GR_IMG_RED] = r

    # Find board edges, spacing, size
    board_edges = find_board(gray, params, res)

    # Find black stones
    black_stones_xy = find_stones(r, params, res, 'B')

    # Find white stones
    white_stones_xy = find_stones(b, params, res, 'W')

    # Convert X-Y coordinates to stone positions
    black_stones = convert_xy(black_stones_xy, res)
    res[grdef.GR_STONES_B] = black_stones
    white_stones = convert_xy(white_stones_xy, res)
    res[grdef.GR_STONES_W] = white_stones

    return res

# Creates a board image for given image shape and board size
# If recognition results are provided, plot them on the board
def generate_board(shape = grdef.DEF_IMG_SIZE, board_size = None, res = None):

    # Prepare params
    if board_size is None:
       if res is None:
          board_size = grdef.DEF_BOARD_SIZE
       else:
          board_size = res[grdef.GR_BOARD_SIZE]

    edges = None
    space_x = None
    space_y = None
    if res is None:
       edges = ((14,14),(shape[grdef.CV_WIDTH]-14, shape[grdef.CV_HEIGTH]-14))
       space_x = (edges[grdef.GR_TO][grdef.GR_X] - edges[grdef.GR_FROM][grdef.GR_X]) / (board_size - 1)
       space_y = (edges[grdef.GR_TO][grdef.GR_Y] - edges[grdef.GR_FROM][grdef.GR_Y]) / (board_size - 1)
    else:
       edges = res[grdef.GR_EDGES]
       space_x, space_y = res[grdef.GR_SPACING]

    # Make up empty image
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:] = grdef.DEF_IMG_COLOR

    # Draw the lines
    for i in range(board_size):
        x1 = int(edges[grdef.GR_FROM][grdef.GR_X] + (i * space_x))
        y1 = int(edges[grdef.GR_FROM][grdef.GR_Y])
        x2 = x1
        y2 = int(edges[grdef.GR_TO][grdef.GR_Y])
        cv2.line(img,(x1,y1),(x2,y2),grdef.COLOR_BLACK,1)

    for i in range(board_size):
        x1 = int(edges[grdef.GR_FROM][grdef.GR_X])
        y1 = int(edges[grdef.GR_FROM][grdef.GR_Y] + (i * space_y))
        x2 = int(edges[grdef.GR_TO][grdef.GR_X])
        y2 = y1
        cv2.line(img, (x1,y1), (x2,y2), grdef.COLOR_BLACK, 1)

    # Draw the stones
    if res is not None:
       black_stones = res.get(grdef.GR_STONES_B)
       white_stones = res.get(grdef.GR_STONES_W)
       r = max(int(min(space_x, space_y) / 2) - 1, 5)

       if black_stones is not None:
          for i in black_stones:
              x1 = int(edges[grdef.GR_FROM][grdef.GR_X] + ((i[2]-1) * space_x))
              y1 = int(edges[grdef.GR_FROM][grdef.GR_Y] + ((i[3]-1) * space_y))
              cv2.circle(img, (x1,y1), r, grdef.COLOR_BLACK, -1)

       if white_stones is not None:
          for i in white_stones:
              x1 = int(edges[grdef.GR_FROM][grdef.GR_X] + ((i[2]-1) * space_x))
              y1 = int(edges[grdef.GR_FROM][grdef.GR_Y] + ((i[3]-1) * space_y))
              cv2.circle(img, (x1,y1), r, grdef.COLOR_BLACK, 1)
              cv2.circle(img, (x1,y1), r-1, grdef.COLOR_WHITE, -1)

    return img

