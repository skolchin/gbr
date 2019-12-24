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
        self.v = np.array(d[0:GR_R+1]).copy()
        self.v.resize(GR_R+1, refcheck=False)
        self.bw = bw
        self.def_v = self.v.copy()
        self.def_bw = bw
        self.forced = False

    @property
    def x(self):
        return self.v[GR_X]

    @x.setter
    def x(self, v):
        self.v[GR_X] = x

    @property
    def y(self):
        return self.v[GR_Y]

    @y.setter
    def y(self, v):
        self.v[GR_Y] = v

    @property
    def a(self):
        return self.v[GR_A]

    @a.setter
    def a(self, v):
        self.v[GR_A] = v

    @property
    def b(self):
        return self.v[GR_B]

    @b.setter
    def b(self, v):
        self.v[GR_B] = v

    @property
    def r(self):
        return self.v[GR_R]

    @r.setter
    def r(self, v):
        self.v[GR_R] = v

    @property
    def pos(self):
        return format_stone_pos(self.v, axis = None)

    def __str__(self):
        return self.pos

    def set(self, stone, bw = None):
        if isinstance(stone, GrStone):
            self.v = stone.v
            self.bw = stone.bw
        elif type(stone) is np.ndarray or type(stone) is list or type(stone) is tuple:
            self.v = np.array(stone).copy()
            self.v.resize(GR_R+1, refcheck=False)
            self.bw = bw if bw is not None else self.bw
        else:
            raise ValueError("Don't know how to handle type " + str(type(stone)))

    def tolist(self):
        r = [int(x) for x in self.v]
        r.extend(self.bw)
        return r

    def full_list(self):
        return [
            [int(x) for x in self.v],
            self.bw,
            [int(x) for x in self.def_v],
            self.def_bw]

# Stone collection
class GrStones(object):
    def __init__(self, stones = None, bw = None):
        self.__stones = dict()
        if stones is not None:
            if bw is None:
                raise ValueError("Stone color not specified")
            self.assign(stones, bw)

    @property
    def stones(self):
        """Stones collection (dictionary with position as a key)"""
        return self.__stones

    @stones.setter
    def stones(self, value):
        """Save stones from another collection"""
        self.assign(value, True)

    @property
    def black(self):
        return np.array([self.__stones[k].v for k in self.__stones if self.__stones[k].bw == STONE_BLACK])

    @property
    def white(self):
        return np.array([self.__stones[k].v for k in self.__stones if self.__stones[k].bw == STONE_WHITE])

    def keys(self):
        """List of positions on board"""
        return self.__stones.keys()

    def forced_stones(self):
        """List of stones with parameters forcefully changed"""
        return [k for k in self.__stones if self.__stones[k].forced]

    def unforced_stones(self):
        """List of stones with none forced changes"""
        return [k for k in self.__stones if not self.__stones[k].forced]

    def changed_stones(self):
        """List of stones with parameters changed since detection"""
        return [k for k in self.__stones
            if self.__stones[k].bw != self.__stones[k].def_bw or \
            not all(np.equal(self.__stones[k].v, self.__stones[k].def_v))]

    def relocated_stones(self):
        """List of stones which had changed color """
        return [k for k in self.__stones
            if self.__stones[k].bw != self.__stones[k].def_bw]

    def toarray(self):
        """Represent stone collection as numpy array.
        Note that color information would not be stored in the array"""
        if len(self.__stones) == 0:
            return np.array([])
        else:
            r = np.empty((len(self.__stones), GR_R+1), dtype = np.int)
            for n, k in enumerate(self.__stones):
                r[n] = self.__stones[k].v
            return r

    def tolist(self):
        """Represent all stones as a list"""
        r = []
        for k in self.__stones:
            r.extend([self.__stones[k].tolist()])
        return r

    def todict(self):
        """Represent all stones as a dictonary"""
        r = dict()
        for k in self.__stones:
            d = self.__stones[k].tolist()
            r[k] = d
        return r

    def add(self, new_stones, bw = None, with_forced = True, set_forced = None):
        """Add or replace stones in collection"""
        if new_stones is None:
            return

        if type(new_stones) is GrStones:
            for k in new_stones:
                p = self.__stones.get(k)
                if p is None:
                    p = new_stones.stones[k]
                    if set_forced is not None: p.forced = set_forced
                    self.__stones[p.pos] = p
                elif not p.forced or with_forced:
                    p.set(new_stones.stones[k])
                    if set_forced is not None: p.forced = set_forced

        elif type(new_stones) is dict:
            if bw is None:
                raise Exception("Stone color required")
            for k in new_stones:
                p = self.__stones.get(k)
                if p is None:
                    n = GrStone(new_stones[k], bw)
                    if set_forced is not None: n.forced = set_forced
                    self.__stones[n.pos] = n
                elif not p.forced or with_forced:
                    p.set(new_stones[k], bw)
                    if set_forced is not None: p.forced = set_forced

        elif type(new_stones) is list or type(new_stones) is np.ndarray:
            if bw is None:
                raise Exception("Stone color required")
            for s in new_stones:
                if type(s) is not list and type(s) is not tuple and type(s) is not np.ndarray:
                    raise Exception("Invalid record type " + str(type(s)))

                n = GrStone(s, bw)
                p = self.__stones.get(n.pos)
                if p is None:
                    if set_forced is not None: n.forced = set_forced
                    self.__stones[n.pos] = n
                elif not p.forced or with_forced:
                    p.set(n)
                    if set_forced is not None: p.forced = set_forced
        else:
            raise Exception("Invalid stone list type " + str(type(new_stones)))

    def remove(self, stone):
        """Remove a stone from collection"""
        p = None
        if isinstance(stone, GrStone):
            p = stone.pos
        elif type(p) is list or type(p) is np.ndarray:
            p = format_stone_pos(stone)
        else:
            p = str(stone)

        if p in self.__stones: del self.__stones[p]

    def assign(self, new_stones, bw = None, with_forced = False):
        """Store new stones in collection.

        Parameters:
            new_stones  dict, GrStones or any iterable
            bw          Stone color, has to be set when iterable or dict is provided
            with_forced Set to True to clear or keep forced stones unchanged
        """
        self.clear(with_forced)
        if not new_stones is None: self.add(new_stones, bw, with_forced = with_forced)


    def clear(self, with_forced = False):
        """Clear collection. If with_forced is False, forced stones remain"""
        if with_forced:
            self.__stones.clear()
        else:
            for k in self.unforced_stones(): del self.__stones[k]

    def reset(self):
        """Reset stone parameters to initial values clearing forced flag"""
        for k in self.__stones:
            s = self.__stones[k]
            s.v = s.def_v.copy()
            s.bw = s.def_bw
            s.forced = False

    def forced_tolist(self):
        """Get a list of forced stones (all parameters).
        Returns a list of stones where each entry contains:
            Current stone parameters (list of ints)
            Current stone color (str)
            Default stone parameters (list of ints)
            Default stone color"""
        r = []
        for k in self.__stones:
            if self.__stones[k].forced:
                r.extend([self.__stones[k].full_list()])
        return r

    def forced_fromlist(self, forced_list):
        """Store forced stones in collection"""
        if forced_list is None:
            return

        for f in forced_list:
            if len(f) < 4:
                raise ValueError("Invalid forced stone list format")
            new_stone = GrStone(f[0], f[1])
            new_stone.def_v = np.array(f[2])
            new_stone.def_bw = f[3]
            new_stone.forced = True
            self.__stones.setdefault(new_stone.pos, new_stone)

    def get(self, key):
        """Returns a stone data for given position of None if it doesn't exist"""
        return self.__stones[key].v if key in self.__stones else None

    def get_stone(self, key = None, stone = None):
        """Returns a stone object for given position of None if it doesn't exist"""
        if key is None:
            if stone is None:
                raise ValueError("Eiher stone or position has to be provided")
            key = format_stone_pos(stone)
        return self.__stones[key] if key in self.__stones else None

    def __iter__(self):
        """Iterator"""
        yield from self.__stones

    def __getitem__(self, key):
        """Getter"""
        return self.__stones[key].v

    def __setitem__(self, key, value):
        """Setter"""
        self.__stones[key].v = value

    def __delitem__(self, key):
        """Deleter"""
        del self.__stones[key]

    def __contains__(self, item):
        """in operation support"""
        return item in self.__stones

    def __str__(self):
        """Printing support"""
        return str(self.todict())

    def __len__(self):
        return len(self.__stones)

    def __array__(self):
        return self.toarray()

    def find_coord(self, x, y):
        """Find a stone at given (X,Y) coordinates.
        If stone found returns tuple(stone properties (list of ints), stone color) otherwise - None"""
        for k in self.__stones:
            s = self.__stones[k].v
            min_x = max(1, int(s[GR_X]) - int(s[GR_R]))
            min_y = max(1, int(s[GR_Y]) - int(s[GR_R]))
            max_x = s[GR_X] + s[GR_R]
            max_y = s[GR_Y] + s[GR_R]
            if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
                return s, self.__stones[k].bw
        return (None, None)

    def find_position(self, a, b):
        """Find a stone at given (A,B) position.
        If stone found returns tuple(stone properties (list of ints), stone color) otherwise - None"""
        s = self.__stones.get(format_stone_pos([0,0,a,b], axis = None))
        return (s.v, s.bw) if s is not None else (None, None)

    def find(self, key):
        """Returns a stone at position specified by key.
        If stone found returns tuple(stone properties (list of ints), stone color) otherwise - None"""
        s = self.__stones.get(key)
        return (s.v, s.bw) if s is not None else (None, None)

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
        stones = self.tolist()
        for s in stones:
            if not(s[GR_A] == p[0] and s[GR_B] == p[1]) and \
            s[GR_A] in rg_a and s[GR_B] in rg_b:
                r.extend([s])
        return r

