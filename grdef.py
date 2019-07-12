#-------------------------------------------------------------------------------
# Name:        Go board parsing project
# Purpose:     Global definitions and constants
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

# Constants
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0,0, 0)
DEF_IMG_SIZE = (500, 500)
DEF_IMG_COLOR = (80, 145, 210)
DEF_BOARD_SIZE = 19
DEF_AVAIL_SIZES = [9, 13, 19]

# Default parameter values
DEF_GR_PARAMS = {
    "HC_MINDIST": 1,
    "HC_SENSITIVITY": 10,
    "HC_MAXRADIUS": 20,
    "HC_MASK": 4,
    "CANNY_MINVAL": 50,
    "CANNY_MAXVAL": 100,
    "CANNY_APERTURE": 3,
    "HL_RHO": 1,
    "HL_THETA": 6,
    "HL_THRESHOLD": 100,
    "HL_RHO2": 1,
    "HL_THETA2": 5,
    "HL_THRESHOLD2": 6,
    #"HL_DILATE": 0,
    "STONES_THRESHOLD_B": 83,
    "STONES_THRESHOLD_W": 218,
    "STONES_MAXVAL_B": 255,
    "STONES_MAXVAL_W": 255,
    "STONES_DILATE_B": 2,
    "STONES_DILATE_W": 0,
    "STONES_ERODE_B": 0,
    "STONES_ERODE_W": 0
#    "BOARD_SIZE": DEF_BOARD_SIZE
}

# Parameter properties: min, max, change, title
GR_PARAMS_PROP = {
    "HC_MINDIST": (1, 30, True),
    "HC_SENSITIVITY": (1, 40, True),
    "HC_MAXRADIUS": (1, 40, True),
    "HC_MASK": (1, 6, True),
    "CANNY_MINVAL": (1, 255, False),
    "CANNY_MAXVAL": (1, 255, False),
    "CANNY_APERTURE": (3, 7, False),
    "HL_RHO": (1, 5, True),
    "HL_THETA": (1, 90, True),
    "HL_THRESHOLD": (1, 255, True),
    "HL_RHO2": (1, 5, True),
    "HL_THETA2": (1, 90, True),
    "HL_THRESHOLD2": (1, 255, True),
    #"HL_DILATE": (0, 10, True),
    "STONES_THRESHOLD_B": (1, 255, True),
    "STONES_THRESHOLD_W": (1, 255, True),
    "STONES_MAXVAL_B": (1, 255, False),
    "STONES_MAXVAL_W": (1, 255, False),
    "STONES_DILATE_B": (0, 10, True),
    "STONES_DILATE_W": (0, 10, True),
    "STONES_ERODE_B": (0, 10, True),
    "STONES_ERODE_W": (0, 10, True)
    #"BOARD_SIZE": (9, 21, True)
}

# Constants for analysis results
GR_STONES_B = "BS"                  # black stones
GR_STONES_W = "WS"                  # white stones
GR_BOARD_SIZE = "BOARD_SIZE"        # board size
GR_IMG_GRAY = "IMG_GRAY"            # grayed out image
GR_IMG_THRESH_B = "IMG_THRESH_B"    # thresholded black stones image
GR_IMG_THRESH_W = "IMG_THRESH_W"    # thresholded white stones image
GR_IMG_MORPH_B = "IMG_MORPH_B"      # morthed black stones image
GR_IMG_MORPH_W = "IMG_MORPH_W"      # thresholded white stones image
GR_IMG_LINES = "IMG_LINES1"         # generated lines image for 1st pass
GR_IMG_LINES2 = "IMG_LINES2"        # generated lines image for 2nd pass
GR_EDGES = "EDGES"                  # edges array (x,y), (x,y)
GR_SPACING = "SPACES"               # spacing of board net (x,y)
GR_NUM_LINES = "NLIN"               # overall number of lines found
GR_NUM_CROSS_H = "NCROSS_H"         # Number of crosses on horizontal line
GR_NUM_CROSS_W = "NCROSS_W"         # Number of crosses on vertical line
