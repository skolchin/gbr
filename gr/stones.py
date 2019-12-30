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
from .utils import format_stone_pos, stone_pos_from_str

import numpy as np

# A stone class
class GrStone(object):
    """A stone class.
    This class holds one stone parameters as a list with the following indexes:
        GR_X    X position on a board (absolute, edges ignored, 0:image width)
        GR_Y    Y position (0:image height)
        GR_A    Position on horizontal axis (1:board_size, minimum on left side)
        GR_B    Position on vertical axis (1:board size, minimum on bottom)
        GR_R    Stone circle radius
        GR_BW   Color (either STONE_BLACK or STONE_WHITE)
    """
    def __init__(self, stone = None, bw = None):
        self.forced = False
        self.added = False

        self.v, self.def_v = None, None
        if stone is not None:
            self.set(stone, bw)

    @property
    def pos(self):
        """Stone position in text format (A10, B2, etc)"""
        return format_stone_pos(self.v, axis = None)

    def __str__(self):
        return self.pos

    def __iter__(self):
        """Iterator"""
        yield from self.v

    def __getitem__(self, index):
        """Getter"""
        return self.v[index]

    def set(self, stone, bw = None):
        """Assign stone params"""
        if stone is None:
            return
        elif isinstance(stone, GrStone):
            self.v = stone.v
            if self.def_v is None: self.def_v = stone.def_v
        elif type(stone) is np.ndarray or type(stone) is list or type(stone) is tuple:
            if len(stone) <= GR_BW and bw is None:
                raise Exception('Color has to be provided')

            # Coordinates are expected to be integer while other props could be of any type
            self.v = [int(x) for x in stone[0:GR_BW]]
            if len(stone) <= GR_BW:
                self.v.extend([None] * (GR_BW - len(stone) + 1))
                self.v[GR_BW] = bw
            else:
                self.v[GR_BW:] = [x for x in stone[GR_BW:]]
            if self.def_v is None: self.def_v = self.v.copy()
        else:
            raise ValueError("Don't know how to handle type " + str(type(stone)))

    def tolist(self):
        """Return stone parameters as a list"""
        return self.v[0:GR_BW+1] if self.v is not None else None

    def to_fulllist(self):
        return [
            self.v[0:GR_BW],
            self.v[GR_BW],
            self.def_v[0:GR_BW],
            self.def_v[GR_BW],
            self.added] if self.v is not None else None

    def from_fulllist(self, f):
        if len(f) < 4:
            raise ValueError("Invalid forced stone list format")
        self.v = list(f[0])
        self.v.extend([f[1]])
        self.def_v = list(f[2])
        self.def_v.extend([f[3]])
        self.forced = True
        self.added = f[4] if len(f) > 4 else False


# Stone collection
class GrStones(object):
    """A collection of stones on board"""
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
        """Assign stones from another collection"""
        self.assign(value, True)

    @property
    def black(self):
        """List of black stones"""
        return [self.__stones[k].v for k in self.__stones if self.__stones[k][GR_BW] == STONE_BLACK]

    @property
    def white(self):
        """List of white stones"""
        return [self.__stones[k].v for k in self.__stones if self.__stones[k][GR_BW] == STONE_WHITE]

    def keys(self):
        """List of stone positions on board"""
        return self.__stones.keys()

    def forced_stones(self):
        """Positions of stones with parameters forcefully changed or stone added"""
        return [k for k in self.__stones if self.__stones[k].forced]

    def unforced_stones(self):
        """Positions of stone which were not forced"""
        return [k for k in self.__stones if not self.__stones[k].forced]

    def changed_stones(self):
        """Positions of stone with parameters changed since detection"""
        return [k for k in self.__stones
            if self.__stones[k].bw != self.__stones[k].def_bw or \
            not all(np.equal(self.__stones[k].v, self.__stones[k].def_v))]

    def relocated_stones(self):
        """List of stones which had changed color """
        return [k for k in self.__stones
            if self.__stones[k].bw != self.__stones[k].def_bw]

    def added_stones(self):
        """Positions of stone added after detection"""
        return [k for k in self.__stones if self.__stones[k].added]

    def toarray(self):
        """Represent stone collection as numpy array.
        Only integer properties (GR_X:GR_R) are returned, color flags are ommitted"""
        if len(self.__stones) == 0:
            return np.array([])
        else:
            r = np.empty((len(self.__stones), GR_R+1), dtype = np.int)
            for n, k in enumerate(self.__stones):
                r[n] = self.__stones[k].v[0:-1]
            return r

    def tolist(self):
        """List of all stones in collection"""
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

    def add_ext(self, new_stones, bw = None, with_forced = True, mark_forced = False, mark_added = False):
        """Add or replace stones in collection - internal"""
        if new_stones is None:
            return

        if type(new_stones) is GrStones:
            for k in new_stones:
                p = self.__stones.get(k)
                if p is None:
                    p = new_stones.stones[k]
                    self.__stones[p.pos] = p
                    if mark_added: p.added = True
                    if mark_forced: p.forced = True
                else:
                    p.set(new_stones.stones[k])
                    if mark_forced: p.forced = True

        elif type(new_stones) is dict:
            for k in new_stones:
                p = self.__stones.get(k)
                if p is None:
                    n = GrStone(new_stones[k], bw)
                    self.__stones[n.pos] = n
                    if mark_added: p.added = True
                    if mark_forced: p.forced = True
                else:
                    p.set(new_stones[k], bw)
                    if mark_forced: p.forced = True

        elif type(new_stones) is list or type(new_stones) is np.ndarray:
            for s in new_stones:
                if type(s) is not list and type(s) is not tuple and type(s) is not np.ndarray:
                    raise Exception("Invalid record type " + str(type(s)))

                p = GrStone(s, bw)
                t = self.__stones.get(p.pos)
                if t is None:
                    self.__stones[p.pos] = p
                    if mark_added: p.added = True
                    if mark_forced: p.forced = True
                else:
                    t.set(p)
                    if mark_forced: t.forced = True
        else:
            raise Exception("Invalid stone list type " + str(type(new_stones)))

    def add(self, new_stones, bw = None, with_forced = True):
        """Add or replace stones in collection - internal"""
        return self.add_ext(new_stones, bw, with_forced, mark_forced = True, mark_added = True)

    def assign(self, new_stones, bw = None, with_forced = False):
        """Store new stones in collection.

        Parameters:
            new_stones  dict, GrStones or any iterable
            bw          Stone color, has to be set when iterable or dict is provided
            with_forced Set to True to clear or keep forced stones unchanged
        """
        self.clear(with_forced)
        if not new_stones is None:
            self.add_ext(new_stones, bw, with_forced = with_forced, mark_forced = False, mark_added = False)

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

    def clear(self, with_forced = False):
        """Clear collection. If with_forced is False, forced stones remain"""
        if with_forced:
            self.__stones.clear()
        else:
            for k in self.unforced_stones(): del self.__stones[k]

    def reset(self):
        """Reset stone parameters to initial values clearing forced flag
        If stone was added after detection, this stone is removed"""
        to_remove = []
        for k in self.__stones:
            s = self.__stones[k]
            if s.added:
                to_remove.extend([k])
            else:
                s.v = s.def_v.copy()
                s.forced = False
        for k in to_remove:
            del self.__stones[k]

    def forced_tolist(self):
        """Get a list of forced stones (all parameters).
        Returns a list of stones where each entry contains:
            0 Current stone parameters (list of ints)
            1 Current stone color (str)
            2 Default stone parameters (list of ints)
            3 Default stone color
            4 Stone added flag"""
        r = []
        for k in self.__stones:
            if self.__stones[k].forced:
                r.extend([self.__stones[k].to_fulllist()])
        return r

    def forced_fromlist(self, forced_list):
        """Store forced stones in collection. See forced_tolist for list format"""
        if forced_list is None:
            return

        for f in forced_list:
            new_stone = GrStone()
            new_stone.from_fulllist(f)
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

    def get_stone_list(self, keys):
        """Returns a list of stones for given position keys"""
        return [self.__stones[k].v for k in keys if k in self.__stones.keys()]

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
                return s
        return None

    def find_position(self, a, b):
        """Find a stone at given (A,B) position.
        If stone found returns tuple(stone properties (list of ints), stone color) otherwise - None"""
        s = self.__stones.get(format_stone_pos([0,0,a,b], axis = None))
        return s.v if s is not None else None

    def find(self, key):
        """Returns a stone at position specified by key.
        If stone found returns tuple(stone properties (list of ints), stone color) otherwise - None"""
        s = self.__stones.get(key)
        return s.v if s is not None else None

    def find_nearby(self, p, d = 1):
        """Finds all stones near specified position.
        Parameters:
            p   stone position coordinates as (A, B) tuple or position string (A9)
            d   delta
        Return: a list of stones closing around given position
        """
        if p is None:
            return None
        elif type(p) is not tuple and type(p) is not list and type(p) is not np.ndarray:
            # Assume it is a string
            p = stone_pos_from_str(str(p))

        r = []
        rg_a = range(max(p[0]-d,1), p[0]+d+1)
        rg_b = range(max(p[1]-d,1), p[1]+d+1)
        stones = self.tolist()
        for s in stones:
            if not(s[GR_A] == p[0] and s[GR_B] == p[1]) and \
            s[GR_A] in rg_a and s[GR_B] in rg_b:
                r.extend([s])
        return r

