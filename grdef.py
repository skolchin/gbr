#-------------------------------------------------------------------------------
# Name:        Go board recognition project
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
    "CANNY_MINVAL": 50,
    "CANNY_MAXVAL": 100,
    "CANNY_APERTURE": 3,
    "HL_RHO": 1,
    "HL_THETA": 6,
    "HL_THRESHOLD": 100,
    "HL_RHO2": 1,
    "HL_THETA2": 5,
    "HL_THRESHOLD2": 6,
    "HC_MINDIST": 1,
    "HC_MAXRADIUS": 20,
    "HC_SENSITIVITY_B": 10,
    "HC_MASK_B": 4,
    "HC_SENSITIVITY_W": 10,
    "HC_MASK_W": 4,
    "BLUR_MASK_B": 4,
    "BLUR_MASK_W": 4,
    "STONES_THRESHOLD_B": 83,
    "STONES_THRESHOLD_W": 218,
    "STONES_MAXVAL_B": 255,
    "STONES_MAXVAL_W": 255,
    "STONES_DILATE_B": 2,
    "STONES_DILATE_W": 0,
    "STONES_ERODE_B": 0,
    "STONES_ERODE_W": 0
}

# Parameter properties: min, max, change, block
GR_PARAMS_PROP = {
    "CANNY_MINVAL": (1, 255, False, "CANNY"),
    "CANNY_MAXVAL": (1, 255, False, "CANNY"),
    "CANNY_APERTURE": (3, 7, False, "CANNY"),
    "HL_RHO": (1, 5, True, "HL"),
    "HL_THETA": (1, 90, True, "HL"),
    "HL_THRESHOLD": (1, 255, True, "HL"),
    "HL_RHO2": (1, 5, True, "HL"),
    "HL_THETA2": (1, 90, True, "HL"),
    "HL_THRESHOLD2": (1, 255, True, "HL"),
    "HC_MINDIST": (1, 30, False, "HC"),
    "HC_MAXRADIUS": (1, 40, False, "HC"),
    "HC_SENSITIVITY_B": (1, 40, True),
    "HC_MASK_B": (1, 10, True, "BLACK"),
    "BLUR_MASK_B": (2, 10, True, "BLACK"),
    "STONES_THRESHOLD_B": (1, 255, True, "BLACK"),
    "STONES_DILATE_B": (0, 10, True, "BLACK"),
    "STONES_ERODE_B": (0, 10, True, "BLACK"),
    "STONES_MAXVAL_B": (1, 255, False, "WHITE"),
    "BLUR_MASK_W": (2, 10, True, "WHITE"),
    "HC_MASK_W": (1, 10, True, "WHITE"),
    "HC_SENSITIVITY_W": (1, 40, True, "WHITE"),
    "STONES_THRESHOLD_W": (1, 255, True, "WHITE"),
    "STONES_MAXVAL_W": (1, 255, False, "WHITE"),
    "STONES_DILATE_W": (0, 10, True, "WHITE"),
    "STONES_ERODE_W": (0, 10, True, "WHITE")
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
