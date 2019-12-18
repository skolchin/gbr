#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     OpenCV Cascade classifier annotation creation script
#
# Author:      kol
#
# Created:     18.12.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from gr.grdef import *
from gr.board import GrBoard
from gr.utils import get_image_area

import numpy as np
import cv2
from pathlib import Path

positive_dir = "./cc/p/"
negative_dir = "./cc/n/"

def overlap(a, b):
    # from: https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
    x1 = max(min(a[0], a[2]), min(b[0], b[2]))
    y1 = max(min(a[1], a[3]), min(b[1], b[3]))
    x2 = min(max(a[0], a[2]), max(b[0], b[2]))
    y2 = min(max(a[1], a[3]), max(b[1], b[3]))
    return x1 < x2 and y1 < y2

def one_file(file_name):
    # open up
    print(file_name)
    board = GrBoard(file_name)

    # generate positive samples (board stones of particular color)
    pd = Path(positive_dir)
    pd.mkdir(exist_ok = True, parents = True)

    prefix = Path(file_name).stem + "_" + Path(file_name).suffix[1:]

    r_list = []
    covered = []
    f_reg = open(str(pd.joinpath("positives.txt")), "a")
    for bw in ['B', 'W']:
        for n, stone in enumerate(board.stones[bw]):
            x = stone[GR_X]
            y = stone[GR_Y]
            r = stone[GR_R]
            area = [max(x-r-1,0),
                max(y-r-1,0),
                min(x+r+1, board.image.shape[1]),
                min(y+r+1, board.image.shape[0])]

            print("\tPositive: {}, {}, {}".format(x, y, r))
            r_list.extend([r])
            covered.extend([area])

            im = get_image_area(board.image, area)
            fn = pd.joinpath(prefix + "_" + bw.lower() + "_" + str(n).zfill(5) + ".png")
            cv2.imwrite(str(fn), im)
            f_reg.write("{} 1 {} {} {} {} \n".format(str(fn.name), 0, 0, im.shape[1], im.shape[0]))
    f_reg.close()

    # generate negative samples (board areas without stones)
    pn = Path(negative_dir)
    pn.mkdir(exist_ok = True, parents = True)

    d = int(np.median(r_list) * 2)
    area = [0, 0, d-1, d-1]
    n = 0
    f_reg = open(str(pn.joinpath("negatives.txt")), "a")

    while area[3] <= board.image.shape[1]:
        rc_in = [x for x in covered if overlap(area, x)]
        if len(rc_in) == 0:
            print("\tNegative: {}".format(area))
            im = get_image_area(board.image, area)
            fn = pn.joinpath(prefix + "_" + bw.lower() + "_" + str(n).zfill(5) + ".png")
            n += 1
            cv2.imwrite(str(fn), im)
            f_reg.write("{}\n".format(str(fn.name), area))

        area[0] = area[2] + 1
        area[2] = area[0] + d
        if area[2] > board.image.shape[0]:
            area[0] = 0
            area[1] = area[3] + 1
            area[2] = area[0] + d
            area[3] = area[1] + d
    f_reg.close()

one_file("./img/go_board_1.png")
one_file("./img/go_board_2.png")
one_file("./img/go_board_6.png")
one_file("./img/go_board_7.png")
one_file("./img/go_board_19.png")
one_file("./img/go_board_20.png")
one_file("./img/go_board_24.png")

