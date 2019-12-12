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
from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.utils import use_named_args

# Recognition quality metrics
#
#   Metric              Meaning
#   ---                 ---
#
#   Board recognition
#
#   No errors           No errors during board recognition +
#   Board size          Board size is 9, 13, 19, 21 +
#   Grid is on black    Grid lines corresponds to dark pixels
#   Expected size       Expected board size found
#
#   Stones recognition
#
#   Any stone           At least 1 stone of each color found +
#   Not too many        Less than 100 stones of each color found +
#   Same number         Almost the same number of stones for each color
#   Proper position     Stone positions are between 1 and board size +
#   Radius is equal     Stone radiuses are almost equal +
#   Correct color       Color of stone area on board image matches stone color
#   No overlaps         Stones are not overlapping +
#   No duplicates       No stones are at the same position +
#   Watershed is OK     Watershed did not report unmatched peaks +
#   Wiped out           Thresholded image is mostly filled with stone color
#   Expected counts     Expected number of stones found

# Base class
class QualityMetric(object):
    def __init__(self, master):
        self.master = master
        self.log = GrLogger(master)

    def check(self, board):
        """Quality check function.
        Should return value in [0,1] range where 0 is perfect, 1 is worst"""
        raise NotImplementedError("Calling of abstract method")

    @property
    def name(self):
        return str(type(self)).split('.')[2].split("'")[0]


#
# Metric classes
#
class BoardSizeMetric(QualityMetric):
    """Board size verification"""
    def check(self, board):
        if board.board_size in DEF_AVAIL_SIZES:
            # Standard size, ok
            return 0.0
        elif board.board_size == 21 or board.board_size == 17:
            # Not standard but acceptable
            return 0.2
        else:
            # Something else - consider this to be a problem
            return 1.0

class NumberOfStonesMetric(QualityMetric):
    """Any stone, too many stones, same number of stones checks"""
    def check(self, board):
        cb = len(board.black_stones) if board.black_stones is not None else 0
        cw = len(board.white_stones) if board.white_stones is not None else 0
        self.master.log.debug("Number of stones: {} black, {} white".format(cb, cw))

        if cb == 0 or cw == 0 or cb > 100 or cw > 100:
            return 1.0
        else:
            d = max(abs(int(cb) - int(cw)) - 2, 0)
            return min(d, 5.0) / 5.0


class ProperPositionMetric(QualityMetric):
    """Proper stone position check"""
    def check(self, board):
        def f(stones):
            # Collect all stones and check all are inside board's space
            if stones is None:
                return 1.0

            s = [x[GR_A] for x in stones]
            s.extend([x[GR_B] for x in stones])

            return 1.0 if len(s) == 0 or max(s) > board.board_size or min(s) <= 0 else 0.0

        return f(board.black_stones) and f(board.white_stones)

class NoDuplicatesMetric(QualityMetric):
    """No duplicates check"""
    def check(self, board):
        # Merge black and white positions
        stones = []
        if board.black_stones is not None:
            stones = [ [x[GR_A], x[GR_B]] for x in board.black_stones]
        if board.white_stones is not None:
            stones.extend ([ [x[GR_A], x[GR_B]] for x in board.white_stones])
        if len(stones) == 0:
            return 1.0

        # Sort on 1st dimension and calculate duplicates count
        u, c = np.unique(stones, return_counts=True, axis = 0)
        n = sum(c[c > 1]) - len(c[c > 1])
        self.master.log.debug("Number of duplicates {}".format(n))
        if self.master.debug:
            for i, x in enumerate(u):
                if c[i] > 1:
                    self.master.log.debug("Stone {} duplicated {} times".format(x, c[i]))

        return min(n,10.0) / float(min(len(stones),10))

class NormalRadiusMetric(QualityMetric):
    """Radius is about the same"""
    def check(self, board):
        # Merge black and white positions
        r = []
        if board.black_stones is not None:
            r = [ x[GR_R] for x in board.black_stones]
        if board.white_stones is not None:
            r.extend ([ x[GR_R] for x in board.white_stones])
        if len(r) == 0:
            return 1.0

        # Check for outliers
        u_q = np.percentile(r, 75)
        l_q = np.percentile(r, 25)
        IQR = (u_q - l_q) * 1.5
        out_r = [x for x in r if (x < l_q - IQR or x > u_q + IQR)]
        self.master.log.debug("Outliers in stone radius list: {}".format(out_r))
        if len(out_r) > 0:
            return 1.0

        # Calculate SD on array with outliers removed
        rr = [x for x in r if (x >= l_q - IQR and x <= u_q + IQR)]
        sd = np.std(rr) if len(rr) > 0 else 0
        self.master.log.debug("Stone radius standard deviation: {}".format(sd))

        return min(sd / 3.0, 1.0)

        #return (min(len(out_r) / 10, 1) + min(sd / 3, 1)) / 2.0

class NoOverlapsMetric(QualityMetric):
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
                return 0.0

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
        stones = []
        if board.black_stones is not None:
            stones = [ (x[GR_X], x[GR_Y], x[GR_R]) for x in board.black_stones]
        if board.white_stones is not None:
            stones.extend ([ (x[GR_X], x[GR_Y], x[GR_R]) for x in board.white_stones])
        if len(stones) == 0:
            return 1.0

        # Make a list of possible combinations and walk through
        pairs = combinations(stones, 2)
        dups = []
        for v in pairs:
            d = find_d(v[0][:2], v[1][:2])
            a = intersection_area(d, v[0][2], v[1][2])
            if a > 0 and a > circle_square(v[0][2], v[1][2]) * 0.05:
                # Stones are considered overlapping with more than 5% intersection
                self.master.log.debug('Stone {} overlaps {} by {}'.format(v[0], v[1], a))
                dups.extend([v[0], v[1]])

        # Reporting
        if len(dups) > 0:
            self.master.log.debug("Overlapped stones found: {}".format(len(dups)))
##            if self.master.debug:
##                show_stones("Overlapped stones: {}".format(len(dups)),
##                    board.image.shape, dups, random_colors(len(dups)))
            return min(len(dups), 10.0) / 10.0
        else:
            self.master.log.debug("No overlapped stones found")
            return 0.0

class WatershedOkMetric(QualityMetric):
    """Watershed did not report missing stones"""
    def check(self, board):
        ws = [w for w in str(self.master.board_log) if w.find("WARNING: WATERSHED") >= 0]
        return 0.0 if ws is None or len(ws) == 0 else 1.0

class WipedOutMetric(QualityMetric):
    """Morphed image contains no objects"""
    def check(self, board):
        def f(bw, bg):
            # Get the image
            if board.results is None:
                return 1.0

            img = board.results.get("IMG_MORPH_" + bw)
            if img is None or min(img.shape[:2]) == 0:
                return 1.0

            # Count number of pixels of bg color within the image
            # It should be less than 75% of image
            u, c = np.unique(img, return_counts = True)
            n = c[ u == bg ]
            nc = img.shape[0] * img.shape[1]
            self.master.log.debug("{} out of {} pixels are of background color for {}".format(n, nc, bw))

            return 1.0 if n > nc * 0.75 else 0.0

        return f('B', 0) and f('W', 255)

class ExpectedCountMetric(QualityMetric):
    """Expected number of stones found"""
    def check(self, board):
        def f(stones, exp_v):
            if exp_v is None or exp_v == 0:
                return 0.0
            else:
                c = len(stones) if stones is not None else 0
                return 0.0 if c == exp_v else 1.0

        return f(board.black_stones, self.master.expected['B']) and \
               f(board.white_stones, self.master.expected['W'])

# Quality checker class
class BoardOptimizer(object):
    """Quality checking and optimization master class"""

    def __init__(self, board = None, debug = False, echo = False):
        self.log = GrLogger(name = 'qc',
            level = logging.DEBUG if debug else logging.INFO,
            echo = echo)
        self.board_log = GrLogger(level = logging.INFO)

        self.metrics = [
            # Metric class, weight
            (BoardSizeMetric, 0.5),
            (NumberOfStonesMetric, 1),
            (ProperPositionMetric, 1),
            (NoDuplicatesMetric, 1),
            (NormalRadiusMetric, 1),
            (NoOverlapsMetric, 1),
            (WatershedOkMetric, 0.5),
            (WipedOutMetric, 1)
        ]
        self.debug = debug
        self.board = board
        self.expected = {' ': 0, 'B': 0, 'W': 0}

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
        self.board.params.add('QUALITY_CHECK',
            GrParam('QUALITY_CHECK', {"v": 1, "no_copy": True, "no_opt": True}))

        # Process board
        self.board_log.clear()
        r = {}
        self.board.process()
        if self.board_log.errors > 0:
            # Any errors during processing means quality is lowest possible
            self.log.error("Error in board processing {}".format(self.board_log.last_error))
            return 1.0, \
                {"Errors ": self.board_log.errors, "Last error": self.board_log.last_error}

        # Check every metric
        for mc in self.metrics:
            m = mc[0](self)
            self.log.debug("Running metric {}".format(m.name))
            r[m.name]  = [m.check(self.board), mc[1]]

        # Check for local extremums
        self.check_empty_board(r)

        # Summarize
        x = [x[0] for x in r.values()]
        w = [x[1] for x in r.values()]
        q = np.average(x, weights = w)
        return q, r


    def opt_space(self, groups):
        """Generates optimization space for board params"""
        space = []
        for g in groups:
            gp = self.board.params.group_params(g)
            for p in gp:
                if not p.no_opt:
                    space.extend([Integer(p.min_v, p.max_v, name = p.key)])
        return space

    def check_empty_board(self, r):
        """Empty board extremum check"""

        # If a few stones detected on board due to invalid settings, other
        # metrics could be validated as OK thus reporting a high quality - which is incorrect
        # But a board can also be near empty and in this case reporting is correct
        # To control this, we check number of stones and if there are only few of them,
        # decrease weights of stone-related metrics
        cb = len(self.board.black_stones) if self.board.black_stones is not None else 0
        cw = len(self.board.white_stones) if self.board.white_stones is not None else 0

        q_ns = r['NumberOfStonesMetric'][0]
        if cb < 5 or cw < 5 and q_ns == 1.0:
            # Suspicious, update weights of all other stone-checking metrics
            self.log.info("Local extremum found, avoiding")
            r['ProperPositionMetric'][1] = 0.5
            r['NoDuplicatesMetric'][1] = 0.5
            r['NormalRadiusMetric'][1] = 0.5
            r['NoOverlapsMetric'][1] = 0.5
            r['WatershedOkMetric'][1] = 0.2

    def optimize(self, groups = None, n_calls = 100, save = "success", callback = None):
        """Optimization"""
        p_init = self.board.params.todict()
        self.log.debug("Initial parameters:")
        for k in p_init:
            self.log.debug("\t{}: {}".format(k, p_init[k]))

        q_init = self.quality()
        self.log.info("Initial quality: {}".format(q_init[0]))
        self.npass = 0

        g = groups if groups is not None else self.board.params.groups
        space = self.opt_space(groups = g)

        # Enclosed objective function
        @use_named_args(space)
        def objective(**params):
            self.npass += 1
            self.board.params = params
            q, p = self.quality()
            self.log.info("Pass #{}: quality {}, {}".format(self.npass, q, p))
            self.log.debug("Pass parameters:")
            for k in params:
                self.log.debug("\t{}: {}".format(k, params[k]))
            return q

        # Enclosed callback function
        def callback_fun(res):
            if callback is None:
                return False
            else:
                return callback({"res": res, "npass": self.npass})

        self.f_res = gbrt_minimize(objective, space, n_calls = n_calls,
            n_jobs = -1, callback = callback_fun)

        self.log.debug("Parameters after optimization")
        for n, v in enumerate(self.f_res.x):
            self.log.debug("\t{}: {}".format(space[n].name,v))
            self.board.params[space[n].name] = v

        self.log.debug("Parameters changed")
        p_res = self.board.params.todict()
        for k in p_init:
            if p_init[k] != p_res[k]:
                self.log.debug("\t{}: {} ( was {})".format(k, p_res[k], p_init[k]))

        q_last = self.quality()
        self.log.info("Resulting quality: {} (was {}), params: {}".format(q_last[0], q_init[0], q_last[1]))

        if q_init <= q_last:
            if save == "always":
                self.log.info("Parameters file {} updated".format(self.board.save_params()))
        else:
            if save == "success" or save == "always":
                self.log.info("Parameters file {} updated".format(self.board.save_params()))

        return q_init, q_last
