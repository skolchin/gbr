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
from .grdef import *
import numpy as np

# List if parameter groups
GROUP_BOARD = '.'
GROUP_BLACK = STONE_BLACK
GROUP_WHITE = STONE_WHITE

GR_PARAM_GROUPS = [
    {"t": "Board recognition", "g": GROUP_BOARD},
    {"t": "Black stones detection", "g": GROUP_BLACK},
    {"t": "White stones detection", "g": GROUP_WHITE}
]

# Parameter default values ans description
GR_PARAMS = {
    # Old params
    "CANNY_MINVAL": {"v": 50, "min_v": 1, "max_v": 255, "no_copy": True},       # Canny min value - cannot be changed
    "CANNY_MAXVAL": {"v": 100, "min_v": 1, "max_v": 255, "no_copy": True},      # Canny max value - cannot be changed
    "CANNY_APERTURE": {"v": 3, "min_v": 3, "max_v": 7, "no_copy": True},        # Canny aperture - cannot be changed
    "HL_RHO":       {"v": 1, "min_v": 1, "max_v": 5, "no_copy": True},          # HoughLines rho - cannot be changed
    "HL_THETA":     {"v": 90, "min_v": 1, "max_v": 90, "no_copy": True},        # HoughLines theta - cannot be changed
    "HL_THRESHOLD": {"v": 0, "min_v": 0, "max_v": 255, "no_copy": True},        # HoughLines threshold - cannot be changed
    "HL_MINLEN":    {"v": 0, "min_v": 0, "max_v": 30, "no_copy": True},         # HoughLines min len - cannot be changed
    "HL_RHO2":      {"v": 1, "min_v": 1, "max_v": 5, "no_copy": True},          # HoughLinesP threshold - cannot be changed
    "HC_MINDIST":   {"v": 1, "min_v": 1, "max_v": 5, "no_copy": True},          # HoughCircles min distance - not used
    "HC_MAXRADIUS": {"v": 20, "min_v": 1, "max_v": 40, "no_copy": True},        # HoughCircles max radius - not used

    # Board params group
    'BOARD_SIZE': {"v": 19, "min_v": 9, "max_v": 21, "g": GROUP_BOARD,
        "title": "Board size", "n": 1, "no_opt": True},                         # Board size
    "HL_THETA2": {"v": 6, "min_v": 1, "max_v": 90, "g": GROUP_BOARD,
        "title": "Angle", "n": 2},                                              # HoughLinesP theta
    "HL_THRESHOLD2": {"v": 40, "min_v": 1, "max_v": 255, "g": GROUP_BOARD,
        "title": "Threshold", "n": 3},                                          # HoughLinesP threshold
    'LUM_EQ': {"v": 0, "min_v": 0, "max_v": 1, "g": GROUP_BOARD,
        "title": "Luminosity filter", "n": 4},                                  # CLAHE filter on/off

    # Black stones detection
    "STONES_THRESHOLD_B": {"v": 84, "min_v": 1, "max_v": 255, "g": GROUP_BLACK,
        "title": "Threshold", "n": 1, "opt_maxv": 150},                         # Threshold
    "HC_SENSITIVITY_B": {"v": 10, "min_v": 1, "max_v": 20, "g": GROUP_BLACK,
        "title": "Sensitivity", "n": 2, "opt_minv": 8, "opt_maxv": 14},         # Sensistivity
    "HC_MASK_B": {"v": 3, "min_v": 1, "max_v": 5, "g": GROUP_BLACK,
        "title": "Mask granularity", "n": 3, "opt_maxv": 4},                    # Mask
    "BLUR_MASK_B": {"v": 0, "min_v": 0, "max_v": 10, "g": GROUP_BLACK,
        "title": "Blurring", "n": 4, "opt_maxv": 4},                            # Blurring
    "STONES_DILATE_B": {"v": 1, "min_v": 0, "max_v": 5, "g": GROUP_BLACK,
        "title": "Dilation", "n": 5, "opt_maxv": 4},                            # Dilation
    "STONES_ERODE_B": {"v": 1, "min_v": 0, "max_v": 5, "g": GROUP_BLACK,
        "title": "Erosion", "n": 6, "opt_maxv": 4},                             # Erosion
    "WATERSHED_B": {"v": 85, "min_v": 0, "max_v": 255, "g": GROUP_BLACK,
        "title": "Watershed threshold", "n": 7, "opt_maxv": 150},               # Watershed
    "WS_MORPH_B": {"v": 0, "min_v": 0, "max_v": 5, "g": GROUP_BLACK,
        "title": "Watershed morphing", "n": 8, "opt_maxv": 3},                  # WS morphing
    "PYRAMID_B": {"v": 0, "min_v": 0, "max_v": 1, "g": GROUP_BLACK,
        "title": "Pyramid filter", "n": 9, "no_opt" : True},                    # Image pyramid filter on/off
    "STONES_MAXVAL_B": {"v": 255, "min_v": 0, "max_v": 255,
        "no_copy": True, "no_opt": True},                                       # MaxVal - cannot be changed

    # White stones detection
    "STONES_THRESHOLD_W": {"v": 173, "min_v": 1, "max_v": 255, "g": GROUP_WHITE,
        "title": "Threshold", "n": 1, "opt_minv": 120},                         # Threshold
    "HC_SENSITIVITY_W": {"v": 10, "min_v": 1, "max_v": 20, "g": GROUP_WHITE,
        "title": "Sensitivity", "n": 2, "opt_minv": 8, "opt_maxv": 14},         # Sensistivity
    "HC_MASK_W": {"v": 3, "min_v": 1, "max_v": 5, "g": GROUP_WHITE,
        "title": "Mask granularity", "n": 3, "opt_maxv": 4},                    # Mask
    "BLUR_MASK_W": {"v": 0, "min_v": 0, "max_v": 10, "g": GROUP_WHITE,
        "title": "Blurring", "n": 4, "opt_maxv": 4},                            # Blurring
    "STONES_DILATE_W": {"v": 1, "min_v": 0, "max_v": 5, "g": GROUP_WHITE,
        "title": "Dilation", "n": 5, "opt_maxv": 4},                            # Dilation
    "STONES_ERODE_W": {"v": 0, "min_v": 0, "max_v": 5, "g": GROUP_WHITE,
        "title": "Erosion", "n": 6, "opt_maxv": 4},                             # Erosion
    "WATERSHED_W": {"v": 131, "min_v": 0, "max_v": 255,
        "g": GROUP_WHITE, "title": "Watershed threshold", "n": 7},                      # Watershed
    "WS_MORPH_W": {"v": 0, "min_v": 0, "max_v": 5, "g": GROUP_WHITE,
        "title": "Watershed morphing", "n": 8, "opt_maxv": 3},                  # WS morphing
    "PYRAMID_W": {"v": 0, "min_v": 0, "max_v": 1, "g": GROUP_WHITE,
        "title": "Pyramid filter", "n": 9, "no_opt": True},                     # Image pyramid filter on/off
    "STONES_MAXVAL_W": {"v": 255, "min_v": 0, "max_v": 255,
        "no_copy": True, "no_opt": True},                                       # MaxVal - cannot be changed

    # no_copy params
    'AREA_MASK': {"no_copy": True, "no_opt": True},
    'TRANSFORM': {"no_copy": True, "no_opt": True},
    'BOARD_EDGES': {"no_copy": True, "no_opt": True},
    'FORCED_STONES': {"no_copy": True, "no_opt": True}
}

# Default parameter value
# Should always contain all parameter fields
GR_PARAMS_DEF = {"v": None, "min_v": None, "max_v": None,
        "g": None, "t": None, "n": None, "no_copy": False, "no_opt": False,
        "opt_minv": None, "opt_maxv": None}

# GrParam class
class GrParam(object):
    """Parameter object"""
    def __init__(self, key, d):
        """Constructor"""

        # Copy all values from default descriptor and then update it with provided dict
        self.key = key
        self.__dict__.update(GR_PARAMS_DEF)
        self.__dict__.update(d)

        # Save initial value
        self.def_v = self.v

    def tolist(self):
        """Represents parameter as a list containing valuable fields"""
        return [self.v, self.min_v, self.max_v, self.g, self.t, self.n]

    def __str__(self):
        """Printing support"""
        return str(self.__dict__)

# Collection of params
class GrParams(object):
    """Parameters collection.
    This class is desiged to be much like an ordinary dictionary of parameters.
    To get a parameter value, use [] or get() methods, but if a parameter object
    is needed, use params collection.
    todict() method returns ordinary dictionary of parameter keys and values.
    """

    def __init__(self, descr = GR_PARAMS):
        """Constructor"""
        self.__params = dict()
        for k in descr:
            self.__params[k] = GrParam(k, descr[k])

    @property
    def params(self):
        """Collection of parameter objects where keys are param names"""
        return self.__params

    @property
    def groups(self):
        """List of group titles"""
        return [g["t"] for g in GR_PARAM_GROUPS]

    def group_params(self, group):
        """List of parameters belonging to specified group.
        A group could be either integer index in GR_GROUPS list, a key
        (g) or a group title (t).
        """
        if type(group) is int or type(group) is np.int or type(group) is np.int32:
            g = GR_PARAM_GROUPS[group]["g"]
        else:
            g = [x["g"] for x in GR_PARAM_GROUPS if x["t"] == group]
            g = g[0] if len(g) > 0 else group

        p = [self.__params[k] for k in self.__params \
            if self.__params[k].g == g and self.__params[k].title is not None]
        return sorted(p, key = lambda k: k.n)

    def keys(self):
        """All parameter keys"""
        return self.__params.keys()

    def get(self, key):
        """Returns a parameter value of None if it doesn't exist"""
        return self.__params[key].v if key in self.__params else None

    def __iter__(self):
        """Iterator"""
        yield from self.__params

    def __getitem__(self, key):
        """Getter"""
        return self.__params[key].v

    def __setitem__(self, key, value):
        """Setter"""
        if not key in self.__params: raise KeyError("Key '" + key + "' not found")
        self.__params[key].v = value

    def __delitem__(self, key):
        """Deleter"""
        del self.__params[key]

    def __contains__(self, item):
        """in operation support"""
        return item in self.__params

    def __str__(self):
        """Printing support"""
        return str(self.todict())

    def add(self, key, param):
        """Add or replace a parameter object"""
        self.__params[key] = param

    def todict(self):
        """Cast to a dictionary of parameter names and values"""
        d = {k: self.__params[k].v for k in self.__params}
        for k in d:
            if type(d[k]) is np.int32: d[k] = int(d[k])
        return d

    def reset(self):
        """Reset all parameters to default values"""
        for p in self.__params:
            self.__params[p].v  = self.__params[p].def_v

    def assign(self, p, copy_all = False):
        """Copy parameters from another dictionary or GrParams collection.
        By default, only those parameters where no_copy = False are copied.
        In order to copy all parameters, pass copy_all = True"""
        for k in self.__params:
            p0 = self.__params[k]
            if (copy_all or p0.no_copy == False) and k in p:
                p0.v = p[k]


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

