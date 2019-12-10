#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Board recognition parameters defintion and processing classes
#
# Author:      kol
#
# Created:     10.12.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

# Constants
COLOR_WHITE = (255, 255, 255)     # white
COLOR_BLACK = (0, 0, 0)           # black
COLOR_RED = (0, 0, 255)           # red
COLOR_BLUE = (255, 0, 0)          # blue
DEF_IMG_SIZE = (500, 500)         # default shape for board generation
MIN_EDGE_DIST = 2                 # minimum board area distance from edge
DEF_IMG_COLOR = (80, 145, 210)    # default generated board color
DEF_BOARD_SIZE = 19               # default board size
DEF_AVAIL_SIZES = [9, 13, 19]     # available standard board sizes
MAX_BOARD_SIZE = 21               # maximum board size
CV_HEIGTH = 0                     # index of height dimension of OpenCv image
CV_WIDTH = 1                      # index of width dimension of OpenCv image
CV_CHANNEL = 2                    # index of channel dimension of OpenCv image
GR_X = 0                          # index of X coordinate dimension
GR_Y = 1                          # index of Y coordinate dimension
GR_A = 2                          # index of horizontal board position dimension
GR_B = 3                          # index of vertical board position dimension
GR_R = 4                          # index of stone radius dimension
GR_FROM = 0                       # index of line start in lines array
GR_TO = 1                         # index of line end in lines array


# Parameter default values ans description
# def, min, max, group, title, order, no_copy (optional)
GR_PARAMS = {
    "CANNY_MINVAL": (50, 1, 255, None, None, None, None),       # Canny min value - cannot be changed
    "CANNY_MAXVAL": (100, 1, 255, None, None, None, None),      # Canny max value - cannot be changed
    "CANNY_APERTURE": (3, 3, 7, None, None, None, None),        # Canny aperture - cannot be changed
    "HL_RHO": (1, 1, 5, None, None, None, None),                # HoughLines rho - cannot be changed
    "HL_THETA": (90, 1, 90, None, None, None, None),            # HoughLines theta - cannot be changed
    "HL_THRESHOLD": (0, 0, 255, None, None, None, None),        # HoughLines threshold - cannot be changed
    "HL_MINLEN": (0, 0, 30, None, None, None, None),            # HoughLines min len - cannot be changed
    "HL_RHO2": (1, 1, 5, None, None, None, None),               # HoughLinesP threshold - cannot be changed
    "HC_MINDIST": (1, 1, 5, None, None, None, None),            # HoughCircles min distance - not used
    "HC_MAXRADIUS": (20, 1, 40, None, None, None, None),        # HoughCircles max radius - not used

    # Board params group
    'BOARD_SIZE': (19, 9, 21, " Board recognition", "Board size", 1),       # Board size
    "HL_THETA2": (90, 1, 90, " Board recognition", "Angle", 2),             # HoughLinesP theta
    "HL_THRESHOLD2": (60, 1, 255, " Board recognition", "Threshold", 3),    # HoughLinesP threshold
    'LUM_EQ': (0, 0, 1, " Board recognition", "Luminosity filter", 4),      # CLAHE filter on/off

    # Black stones detection
    "STONES_THRESHOLD_B": (85, 1, 255, "Black stones detection", "Threshold", 1),   # Threshold
    "HC_SENSITIVITY_B": (10, 1, 40, "Black stones detection", "Sensitivity", 2),    # Sensistivity
    "HC_MASK_B": (3, 1, 10, "Black stones detection", "Mask granularity", 3),       # Mask
    "BLUR_MASK_B": (0, 0, 10, "Black stones detection", "Blurring", 4),             # Blurring
    "STONES_DILATE_B": (1, 0, 10, "Black stones detection", "Dilation", 5),         # Dilation
    "STONES_ERODE_B": (1, 0, 10, "Black stones detection", "Erosion", 6),           # Erosion
    "WATERSHED_B": (90, 0, 255, "Black stones detection", "Watershed", 7),          # Watershed
    "WS_MORPH_B": (0, 0, 10, "Black stones detection", "Watershed morphing", 8),    # WS morphing
    "PYRAMID_B": (0, 0, 1, "Black stones detection", "Pyramid filter", 9),          # Image pyramid filter on/off
    "STONES_MAXVAL_B": (255, 0, 255, None, None, None, None),                       # MaxVal - cannot be changed

    # White stones detection
    "STONES_THRESHOLD_W": (150, 1, 255, "White stones detection", "Threshold", 1),  # Threshold
    "HC_SENSITIVITY_W": (10, 1, 40, "White stones detection", "Sensitivity", 2),    # Sensistivity
    "HC_MASK_W": (3, 1, 10, "White stones detection", "Mask granularity", 3),       # Mask
    "BLUR_MASK_W": (0, 0, 10, "White stones detection", "Blurring", 4),             # Blurring
    "STONES_DILATE_W": (1, 0, 10, "White stones detection", "Dilation", 5),         # Dilation
    "STONES_ERODE_W": (1, 0, 10, "White stones detection", "Erosion", 6),           # Erosion
    "WATERSHED_W": (150, 0, 255, "White stones detection", "Watershed", 7),         # Watershed
    "WS_MORPH_W": (0, 0, 10, "White stones detection", "Watershed morphing", 8),    # WS morphing
    "PYRAMID_W": (0, 0, 1, "White stones detection", "Pyramid filter", 9),          # Image pyramid filter on/off
    "STONES_MAXVAL_W": (255, 0, 255, None, None, None, None),                       # MaxVal - cannot be changed

    # no_copy params
    'AREA_MASK': (None, None, None, None, None, None, None, True),
    'TRANSFORM': (None, None, None, None, None, None, None, True),
    'BOARD_EDGES': (None, None, None, None, None, None, None, True)
}


# GrParam class
class GrParam(object):
    def __init__(self, *args):
        if len(args) < 7:
            raise ValueError("Not enought parameters for '" + args[0] + "'")

        self.key = args[0]
        self.v = args[1]
        self.min_v = args[2]
        self.max_v = args[3]
        self.group = args[4]
        self.title = args[5]
        self.order = args[6]

        self.no_copy = False
        if len(args) > 7: self.no_copy = args[7]

        self.def_v = self.v

    def tolist(self):
        return [self.v, self.min_v, self.max_v, self.group, self.title, self.order]

# Collection of params
class GrParams(object):
    def __init__(self, descr = GR_PARAMS):
        self.__params = dict()
        for k in descr:
            self.__params[k] = GrParam(k, *descr[k])

    @property
    def params(self):
        return self.__params

    @property
    def keys(self):
        return self.__params.keys()

    @property
    def groups(self):
        return sorted(set([self.__params[k].group for k in self.__params if self.__params[k].group is not None]))

    def group_params(self, group):
        p = [self.__params[k] for k in self.__params \
            if self.__params[k].group == group and self.__params[k].title is not None]
        return sorted(p, key = lambda k: k.order)

    def get(self, key):
        return self.__params[key].v if key in self.__params else None

    def __getitem__(self, key):
        return self.__params[key].v

    def __setitem__(self, key, value):
        if not key in self.__params: raise KeyError("Key '" + key + "' not found")
        self.__params[key].v = value

    def __contains__(self, item):
        return item in self.__params

    def __str__(self):
        return str(self.todict())

    def add(self, key, param):
        self.__params[key] = param

    def todict(self):
        return {k: self.__params[k].v for k in self.__params}

    def reset(self):
        for p in self.__params:
            self.__params[p].v  = self.__params[p].def_v

    def assign(self, p, copy_all = False):
        for k in self.__params:
            if (not self.__params[k].no_copy or copy_all) and k in p:
                self.__params[k].v = p[k]

# Analysis results
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
GR_IMG_EDGES = "IMG_EDGES"          # edges image
GR_EDGES = "EDGES"                  # edges array (x,y), (x,y)
GR_SPACING = "SPACES"               # spacing of board net (x,y)
GR_NUM_LINES = "NLIN"               # overall number of lines found
GR_NUM_CROSS_H = "NCROSS_H"         # Number of crosses on horizontal line
GR_NUM_CROSS_W = "NCROSS_W"         # Number of crosses on vertical line
GR_IMAGE_SIZE = "IMAGE_SIZE"        # Image size (width, height)
GR_IMG_WS_B = "IMG_WATERSHED_B"     # Watershed black stones image
GR_IMG_WS_W = "IMG_WATERSHED_W"     # Watershed white stones image

