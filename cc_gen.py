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
from random import randrange

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
    # Open board
    print(file_name)
    board = GrBoard(file_name)
    prefix = Path(file_name).stem + "_" + Path(file_name).suffix[1:]
    r_list = []
    covered = []

    # Generate positive samples (board stones)
    pd = Path(positive_dir)
    pd.mkdir(exist_ok = True, parents = True)
    f_reg = open(str(pd.joinpath("positives.txt")), "a")
    ns = 0
    for stone in board.all_stones:
        x, y, a, b, r, bw = stone

        # Check no stones around
        nearby_stones = board.find_nearby(stone, 1)
        if len(nearby_stones) > 0:
            # Stones too close, just save area to exclude
            area = [max(x-r-1,0),
                max(y-r-1,0),
                min(x+r+1, board.image.shape[CV_WIDTH]),
                min(y+r+1, board.image.shape[CV_HEIGTH])]
            covered.extend([area])
        else:
            # Suitable stone, save positive image
            area = [max(x-r-5,0),
                max(y-r-5,0),
                min(x+r+5, board.image.shape[CV_WIDTH]),
                min(y+r+5, board.image.shape[CV_HEIGTH])]

            print("\tPositive: {}, {}, {}".format(x, y, r))
            r_list.extend([r])
            covered.extend([area])

            im = get_image_area(board.image, area)
            fn = pd.joinpath(prefix + "_" + str(ns).zfill(5) + ".png")
            cv2.imwrite(str(fn), im)
            f_reg.write("{} 1 {} {} {} {} \n".format(str(fn.name), 0, 0, im.shape[1], im.shape[0]))
            ns += 1
    f_reg.close()

    # Generate negative samples (board areas without stones)
    pn = Path(negative_dir)
    pn.mkdir(exist_ok = True, parents = True)

    # Find background color
    u, c =  np.unique(board.image.reshape(-1, board.image.shape[-1]), axis=0, return_counts=True)
    bg_c = u[c.argmax()]

    # Check black or white color selected
    if sum(bg_c) < 40 or sum(bg_c) >= 750:
        cc = c.argsort()
        n = -2
        while sum(bg_c) < 40 or sum(bg_c) >= 750:
            bg_c = u[cc[n]]
            n -= 1

    # Prepare board image with stones removed
    neg_img = board.image.copy()
    for c in covered:
        patch = np.full((c[3]-c[1], c[2]-c[0], neg_img.shape[2]), bg_c, dtype=neg_img.dtype)
        neg_img[c[1]:c[3], c[0]:c[2]] = patch[:]

    fn = pn.joinpath(prefix + "_neg.png")
    cv2.imwrite(str(fn), neg_img)

    # Slice the no-stones image by random pieces generating number of
    # images not less than number of positive images
    w = int(round(neg_img.shape[CV_WIDTH] / 4,0))
    h = int(round(neg_img.shape[CV_HEIGTH] / 4,0))
    nn_max = ns

    f_reg = open(str(pn.joinpath("negatives.txt")), "a")
    for nn in range(nn_max):
        x = randrange(0, neg_img.shape[CV_WIDTH] - w)
        y = randrange(0, neg_img.shape[CV_HEIGTH] - h)

        area = [x, y, x + w, y + h]
        print("\tNegative: {}".format(area))
        im = get_image_area(neg_img, area)

        fn = pn.joinpath(prefix + "_" + str(nn).zfill(5) + ".png")
        cv2.imwrite(str(fn), im)
        f_reg.write("{}\n".format("n/" + str(fn.name)))

    f_reg.close()

##    d = int(np.median(r_list) * 2)
##    area = [0, 0, d-1, d-1]     # x1, y1, x2, y2
##
##    while area[3] <= board.image.shape[CV_HEIGTH]:
##        rc_in = [x for x in covered if overlap(area, x)]
##        if len(rc_in) == 0:
##            print("\tNegative: {}".format(area))
##            im = get_image_area(board.image, area)
##            fn = pn.joinpath(prefix + "_" + bw.lower() + "_" + str(n).zfill(5) + ".png")
##            n += 1
##            cv2.imwrite(str(fn), im)
##            f_reg.write("{}\n".format("n/" + str(fn.name)))
##
##        area[0] = area[2] + 1
##        area[2] = area[0] + d
##        if area[2] > board.image.shape[CV_WIDTH]:
##            area[0] = 0
##            area[1] = area[3] + 1
##            area[2] = area[0] + d
##            area[3] = area[1] + d
##    f_reg.close()

one_file("./img/go_board_1.png")
one_file("./img/go_board_2.png")
one_file("./img/go_board_6.png")
one_file("./img/go_board_7.png")
one_file("./img/go_board_10.png")
one_file("./img/go_board_19.png")
one_file("./img/go_board_20.png")
one_file("./img/go_board_21.png")
one_file("./img/go_board_22.png")
one_file("./img/go_board_24.png")
one_file("./img/go_board_26.png")
one_file("./img/go_board_27.png")
one_file("./img/go_board_53.jpg")
one_file("./img/go_board_54.jpg")
one_file("./img/go_board_55.png")
