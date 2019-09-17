#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Global definitions and constants
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

# Constants
COLOR_WHITE = (255, 255, 255)     # white
COLOR_BLACK = (0,0, 0)            # black
DEF_IMG_SIZE = (500, 500)         # default shape for board generation
DEF_IMG_COLOR = (80, 145, 210)    # default generated board color
DEF_BOARD_SIZE = 19               # default board size
DEF_AVAIL_SIZES = [9, 13, 19]     # available standard board sizes
MAX_BOARD_SIZE = 21               # maximum board size
CV_HEIGTH = 0                     # index of height dimension of OpenCv image
CV_WIDTH = 1                      # index of width dimension of OpenCv image
CV_CHANNEL = 2                    # index of channel dimension of OpenCv image
GR_X = 0                          # index of X coordinate dimension in stones array
GR_Y = 1                          # index of Y coordinate dimension in stones array
GR_A = 2                          # index of X board position dimension in stones array
GR_B = 3                          # index of Y board position dimension in stones array
GR_R = 4                          # index of stone radius dimension in stones array
GR_FROM = 0                       # index of line start in lines array
GR_TO = 1                         # index of line end in lines array

# Default parameter values
DEF_GR_PARAMS = {
    "CANNY_MINVAL": 50,
    "CANNY_MAXVAL": 100,
    "CANNY_APERTURE": 3,
    "HL_RHO": 1,
    "HL_THETA": 90,
    "HL_THRESHOLD": 100,
    "HL_RHO2": 1,
    "HL_THETA2": 90,
    "HL_THRESHOLD2": 6,
    "HC_MINDIST": 1,
    "HC_MAXRADIUS": 20,
    "HC_SENSITIVITY_B": 10,
    "HC_MASK_B": 3,
    "HC_SENSITIVITY_W": 10,
    "HC_MASK_W": 3,
    "BLUR_MASK_B": 0,
    "BLUR_MASK_W": 0,
    "STONES_THRESHOLD_B": 83,
    "STONES_THRESHOLD_W": 150,
    "STONES_MAXVAL_B": 255,
    "STONES_MAXVAL_W": 255,
    "STONES_DILATE_B": 1,
    "STONES_DILATE_W": 0,
    "STONES_ERODE_B": 1,
    "STONES_ERODE_W": 0,
    "WATERSHED_B": 0,
    "WS_MORPH_B": 0,
    "WATERSHED_W": 0,
    "WS_MORPH_W": 0,
    "PYRAMID_B": 0,
    "PYRAMID_W": 0,
}

# Parameter properties: min, max, change, block
GR_PARAMS_PROP = {
    "CANNY_MINVAL": (1, 255, None),
    "CANNY_MAXVAL": (1, 255, None),
    "CANNY_APERTURE": (3, 7, None),

    "HL_RHO": (1, 5, "Lines detection"),
    "HL_THETA": (1, 90, "Lines detection"),
    "HL_THRESHOLD": (1, 255, "Lines detection"),
    "HL_RHO2": (1, 5, "Lines detection"),
    "HL_THETA2": (1, 90, "Lines detection"),
    "HL_THRESHOLD2": (1, 255, "Lines detection"),

    "HC_MINDIST": (1, 30, None),
    "HC_MAXRADIUS": (1, 40, None),

    "HC_SENSITIVITY_B": (1, 40, "Black stones detection"),
    "HC_MASK_B": (1, 10, "Black stones detection"),
    "BLUR_MASK_B": (0, 10, "Black stones detection"),
    "STONES_THRESHOLD_B": (1, 255, "Black stones detection"),
    "STONES_DILATE_B": (0, 10, "Black stones detection"),
    "STONES_ERODE_B": (0, 10, "Black stones detection"),
    "STONES_MAXVAL_B": (1, 255, None),
    "WATERSHED_B": (0, 255, "Black stones detection"),
    "WS_MORPH_B": (0, 10, "Black stones detection"),
    "PYRAMID_B": (0, 1, "Black stones detection"),

    "BLUR_MASK_W": (0, 10, "White stones detection"),
    "HC_MASK_W": (1, 10, "White stones detection"),
    "HC_SENSITIVITY_W": (1, 40, "White stones detection"),
    "STONES_THRESHOLD_W": (1, 255, "White stones detection"),
    "STONES_DILATE_W": (0, 10, "White stones detection"),
    "STONES_ERODE_W": (0, 10, "White stones detection"),
    "STONES_MAXVAL_W": (1, 255, None),
    "WATERSHED_W": (0, 255, "White stones detection"),
    "WS_MORPH_W": (0, 10, "White stones detection"),
    "PYRAMID_W": (0, 1, "White stones detection")
}

# Constants for analysis results
GR_STONES_B = "BS"                  # black stones
GR_STONES_W = "WS"                  # white stones
GR_BOARD_SIZE = "BOARD_SIZE"        # board size
GR_IMG_GRAY = "IMG_GRAY"            # grayed out image
GR_IMG_BLUE = "IMG_CHANNEL_B"       # blue channel of the image
GR_IMG_RED = "IMG_CHANNEL_R"        # red channel of the image
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
GR_IMAGE_SIZE = "IMAGE_SIZE"        # Image size (width, height)
GR_IMG_WS_B = "IMG_WATERSHED_B"     # Watershed black stones image
GR_IMG_WS_W = "IMG_WATERSHED_W"     # Watershed white stones image
