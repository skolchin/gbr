#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     itertools.accumulate() for Python2
#
# Author:      Taken from https://docs.python.org/dev/library/itertools.html#itertools.accumulate
#
# Created:     22-09-2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
def accumulate (iterable, func, initial=None):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element)
        yield total
