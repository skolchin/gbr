#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Recognition validation model
#
# Author:      kol
#
# Created:     06.12.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from .board import GrBoard
from .log import GrLogger
from .grdef import *
from .params import GrParam, GrParams
from .utils import show_stones, random_colors

import cv2
import numpy as np
from itertools import combinations
import logging

from skopt.space.space import Real, Integer

# Recognition quality metrics
#
# Board recognition
#
#   Metric              Meaning
#   ---                 ---
#   No errors           No errors during board recognition +
#   Board size          Board size is 9, 13, 19, 21 +
#   Grid is on black    Grid lines corresponds to dark pixels
#
#   Stones recognition
#
#   Any stone           At least 1 stone of each color found +
#   Not too many        Less than 100 stones of each color found +
#   Proper position     Stone positions are between 1 and board size +
#   Radius is equal     Stone radiuses are almost equal +
#   Correct color       Color of stone image area matches stone colors
#   No overlaps         Stones are not overlapping +
#   No duplicates       No stones are at the same position +
#   Watershed is OK     Watershed did not report unmatched peaks

# Base class
class ParseQualityMetric(object):
    def __init__(self, master):
        self.master = master
        self.log = GrLogger(master)

    def check(self, board):
        raise NotImplementedError("Calling of abstract method")

    @property
    def name(self):
        return str(type(self)).split('.')[2].split("'")[0]

#
# Metric classes
#
class BoardSizeMetric(ParseQualityMetric):
    """Board size verification"""
    def check(self, board):
        if board.board_size in DEF_AVAIL_SIZES:
            # Standard size, ok
            return 0
        elif board.board_size == 21 or board.board_size == 17:
            # Well, it's not standard but acceptable
            return 0.2
        else:
            # Something else - consider this to be a problem
            return 1

class AnyStoneMetric(ParseQualityMetric):
    """Have any stone check"""
    def check(self, board):
        def f(stones):
            return 1 if stones is None or len(stones) == 0 else 0

        return (f(board.black_stones) + f(board.white_stones)) / 2

class TooManyStonesMetric(ParseQualityMetric):
    """Too many stones detected check"""
    def check(self, board):
        def f(stones):
            return 1 if stones is None or len(stones) > 100 else 0

        return (f(board.black_stones) + f(board.white_stones)) / 2

class ProperPositionMetric(ParseQualityMetric):
    """Proper stone position check"""
    def check(self, board):
        def f(stones):
            # Collect all stones and check all are inside board's space
            s = [x[GR_A] for x in stones]
            s.extend([x[GR_B] for x in stones])

            return 1 if len(s) == 0 or max(s) > board.board_size or min(s) <= 0 else 0

        return f(board.black_stones) and f(board.white_stones)

class NoDuplicatesMetric(ParseQualityMetric):
    """No duplicates check"""
    def check(self, board):
        # Merge black and white positions
        stones = [ [x[GR_A], x[GR_B]] for x in board.black_stones]
        stones.extend ([ [x[GR_A], x[GR_B]] for x in board.white_stones])
        if len(stones) == 0:
            return 1

        # Sort on 1st dimension and calculate duplicates count
        u, c = np.unique(stones, return_counts=True, axis = 0)
        n = sum(c[c > 1]) - len(c[c > 1])
        logging.debug("Number of duplicates {}".format(n))
        if self.master.debug:
            for i, x in enumerate(u):
                if c[i] > 1:
                    logging.debug("Stone {} duplicated {} times".format(x, c[i]))

        return min(n,10) / float(min(len(stones),10))

class NormalRadiusMetric(ParseQualityMetric):
    """Radius is about the same"""
    def check(self, board):
        # Merge black and white positions
        r = [ x[GR_R] for x in board.black_stones]
        r.extend ([ x[GR_R] for x in board.white_stones])
        if len(r) == 0:
            return 1

        # Check for outliers
        u_q = np.percentile(r, 75)
        l_q = np.percentile(r, 25)
        IQR = (u_q - l_q) * 1.5
        out_r = [x for x in r if (x < l_q - IQR or x > u_q + IQR)]
        logging.debug("Outliers in stone radius found: {}".format(out_r))

        # Calculate SD on array with outliers removed
        rr = [x for x in r if (x >= l_q - IQR and x <= u_q + IQR)]
        sd = np.std(rr) if len(rr) > 0 else 0
        logging.debug("Stone radius standard deviation: {}".format(sd))

        return (min(len(out_r) / 10, 1) + min(sd / 3, 1)) / 2

class NoOverlapsMetric(ParseQualityMetric):
    """Stones are not overlapping"""

    def check(self, board):
        def circle_square(R, r):
            """Square of minimal of circles"""
            return np.pi * min(np.int32(R), np.int32(r))**2

        def intersection_area(d, R, r):
            """Return the area of intersection of two circles.

            The circles have radii R and r, and their centres are separated by d.
            Source: https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
            """
            d, R, r = np.float(d), np.float(R), np.float(r)
            if d <= abs(R-r):
                # One circle is entirely enclosed in the other
                return circle_square(R, r)
            if d >= r + R:
                # The circles don't overlap at all
                return 0

            r2, R2, d2 = r**2, R**2, d**2
            alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
            beta = np.arccos((d2 + R2 - r2) / (2*d*R))
            return ( r2 * alpha + R2 * beta -
                     0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
                     )

        def find_d(p1, p2):
            """Determine distance among two points"""
            x = np.int32(p1[0]) - np.int32(p2[0])
            y = np.int32(p1[1]) - np.int32(p2[1])
            return np.sqrt(x**2 + y**2)

        # Merge black and white positions
        stones = [ (x[GR_X], x[GR_Y], x[GR_R]) for x in board.black_stones]
        stones.extend ([ (x[GR_X], x[GR_Y], x[GR_R]) for x in board.white_stones])

        # Make a list of possible combinations and walk through
        pairs = combinations(stones, 2)
        dups = []
        for v in pairs:
            d = find_d(v[0][:2], v[1][:2])
            a = intersection_area(d, v[0][2], v[1][2])
            if a > 0 and a > circle_square(v[0][2], v[1][2]) * 0.05:
                # Stones are considered overlapping with more than 5% intersection
                logging.debug('Stone {} overlaps {} by {}'.format(v[0], v[1], a))
                dups.extend([v[0], v[1]])

        # Reporting
        if len(dups) > 0:
            logging.debug("Overlapped stones found: {}".format(len(dups)))
            if self.master.debug:
                show_stones("Overlapped stones: {}".format(len(dups)),
                    board.image.shape, dups, random_colors(len(dups)))
            return min(len(dups), 10) / 10
        else:
            logging.debug("No overlapped stones found")
            return 0

# Quality checker class
class BoardQualityChecker(object):
    """Quality checker master class"""

    def __init__(self, board = None, debug = False):
        self.log = GrLogger(level = logging.DEBUG if debug else logging.CRITICAL)
        self.metrics = [
            (BoardSizeMetric, 0.5),
            (AnyStoneMetric, 0.3),
            (TooManyStonesMetric, 0.5),
            (ProperPositionMetric, 1),
            (NoDuplicatesMetric, 1),
            (NormalRadiusMetric, 1),
            (NoOverlapsMetric, 1)
        ]
        self.debug = debug
        self.board = board

    def quality(self, board = None):
        """Board quality check function"""

        # Load board
        if board is not None:
            if isinstance(board, GrBoard):
                # New board instance given
                self.board = board
            else:
                # Assume this is a file name
                self.board = GrBoard()
                self.board.load_image(str(board), f_process = False)
        if self.board is None:
            raise ValueError("Board is not assigned")
        if self.board.is_gen_board:
            raise ValueError("Board not loaded")

        # Add extra parameter signaling of quality check been run
        self.board.params.add('QUALITY_CHECK', GrParam('QUALITY_CHECK', 1, 0, 1, None, None, None, True))

        # Process board
        self.log.clear()
        r = {}
        self.board.process()
        if self.log.errors > 0:
            # Any errors during processing means quality is lowest possible
            return 1, {"Errors: ", self.log.errors}

        # Check every metric
        for mc in self.metrics:
            m = mc[0](self)
            logging.info("Running metric {}".format(m.name))
            r[m.name]  = (m.check(self.board), mc[1])

        # Summarize
        q = [x[0] * x[1] for x in r.values()]
        return sum(q) / len(q), r


    def opt_space(self, group):
        """Generates optimization space for board params"""
        space = []
        g = self.board.params.group_params(group)
        for p in g:
            if not p.no_copy:
                space.extend([Integer(p.min_v, p.max_v, name = p.key)])
        return space
