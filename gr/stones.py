#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go stone and stone collection class
#
# Author:      kol
#
# Created:     23.12.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from .grdef import *
from .utils import format_stone_pos

import numpy as np

# A stone class
class GrStone(object):
    def __init__(self, d, bw = None):
        self.d = np.array(d).copy()
        self.d.resize(GR_R+1, refcheck=False)
        self.bw = bw
        self.forced = False

    @property
    def x(self):
        return self.d[GR_X]

    @x.setter
    def x(self, v):
        self.d[GR_X] = x

    @property
    def y(self):
        return self.d[GR_Y]

    @y.setter
    def y(self, v):
        self.d[GR_Y] = v

    @property
    def a(self):
        return self.d[GR_A]

    @a.setter
    def a(self, v):
        self.d[GR_A] = v

    @property
    def b(self):
        return self.d[GR_B]

    @b.setter
    def b(self, v):
        self.d[GR_B] = v

    @property
    def r(self):
        return self.d[GR_R]

    @r.setter
    def r(self, v):
        self.d[GR_R] = v

    @property
    def pos(self):
        return format_stone_pos(self.d, axis = None)

    def __str__(self):
        return self.pos

    def tolist(self):
        return list(self.d)

# Stone collection
class GrStones(object):
    def __init__(self, bw, stones = None):
        self.__stones = dict()
        self.bw = bw
        if stones is not None:
            self.assign(stones)

    @property
    def stones(self):
        return self.__stones

    @stones.setter
    def stones(self, value):
        self.assign(value, True)

    def keys(self):
        return self.__stones.keys()

    def forced_stone_keys(self):
        return [k for k in self.__stones if self.__stones[k].forced]

    def unforced_stone_keys(self):
        return [k for k in self.__stones if not self.__stones[k].forced]

    def toarray(self):
        if len(self.__stones) == 0:
            return np.array([])
        else:
            r = np.empty((len(self.__stones), GR_R+1), dtype = np.int)
            for n, k in enumerate(self.__stones):
                r[n] = self.__stones[k].d
            return r

    def tolist(self):
        r = []
        for k in self.__stones:
            r.extend([self.__stones[k].tolist()])
        return r

    def assign(self, new_stones, with_forced = False):
        self.clear(with_forced)
        if new_stones is None:
            return

        if type(new_stones) is dict or type(new_stones) is GrStones:
            for k in new_stones:
                new_stone = GrStone(new_stones[k], self.bw)
                self.__stones.setdefault(new_stone.pos, new_stone)
        else:
            for s in new_stones:
                new_stone = GrStone(s, self.bw)
                self.__stones.setdefault(new_stone.pos, new_stone)

    def clear(self, with_forced = False):
        if with_forced:
            self.__stones.clear()
        else:
            fs = self.unforced_stone_keys()
            for k in fs: del self.__stones[k]

    def get(self, key):
        """Returns a stone for given position of None if it doesn't exist"""
        return self.__stones[key].d if key in self.__stones else None

    def __iter__(self):
        """Iterator"""
        yield from self.__stones

    def __getitem__(self, key):
        """Getter"""
        return self.__stones[key].d

    def __setitem__(self, key, value):
        """Setter"""
        self.__stones[key].d = value

    def __delitem__(self, key):
        """Deleter"""
        del self.__stones[key]

    def __contains__(self, item):
        """in operation support"""
        return item in self.__stones

    def __str__(self):
        """Printing support"""
        return str(self.toarray())

    def __len__(self):
        return len(self.__stones)

    def __array__(self):
        return self.toarray()

    def find_coord(self, x, y):
        """Returns a stone at given (X,Y) coordinates or None"""
        for k in self.__stones:
            s = self.__stones[k].d
            min_x = max(1, int(s[GR_X]) - int(s[GR_R]))
            min_y = max(1, int(s[GR_Y]) - int(s[GR_R]))
            max_x = s[GR_X] + s[GR_R]
            max_y = s[GR_Y] + s[GR_R]
            if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
                return s

        return None

    def find_position(self, a, b):
        """Returns a stone at given (A,B) position or None"""
        s = self.__stones.get(format_stone_pos([0,0,a,b], axis = None))
        return s.d if s is not None else None

    def find(self, key):
        """Returns a stone at position specified by key or None"""
        s = self.__stones.get(key)
        return s.d if s is not None else None


    def find_nearby(self, p, d = 1):
        """Finds all stones near specified position.
        Parameters:
            p   board position coordinates as (A, B) tuple
            d   delta
        Return: a list of stones closing around given position
        """
        if p is None:
            return None
        r = []
        rg_a = range(max(p[0]-d,1), p[0]+d+1)
        rg_b = range(max(p[1]-d,1), p[1]+d+1)
        stones = self.toarray()
        for s in stones:
            if not(s[GR_A] == p[0] and s[GR_B] == p[1]) and \
            s[GR_A] in rg_a and s[GR_B] in rg_b:
                r.extend([s])
        return np.array(r)

    @staticmethod
    def all_stones(black_stones, white_stones):
        r = []
        if not black_stones is None:
            if isinstance(black_stones, GrStones):
                black_stones = black_stones.tolist()
            r.extend([list(r) + [STONE_BLACK] for r in black_stones])
        if not white_stones is None:
            if isinstance(white_stones, GrStones):
                white_stones = white_stones.tolist()
            r.extend([list(r) + [STONE_WHITE] for r in white_stones])
        return r

    @staticmethod
    def find_nearby_all(black_stones, white_stones, p, d):
        if p is None:
            return None

        stones = GrStones.all_stones(black_stones, white_stones)

        r = []
        rg_a = range(max(p[0]-d,1), p[0]+d+1)
        rg_b = range(max(p[1]-d,1), p[1]+d+1)
        for s in stones:
            if not(s[GR_A] == p[0] and s[GR_B] == p[1]) and \
            s[GR_A] in rg_a and s[GR_B] in rg_b:
                r.extend([s])
        return r

