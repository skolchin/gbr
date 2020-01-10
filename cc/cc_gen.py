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
import numpy as np
import cv2
import os
import sys
import glob
from pathlib import Path
from random import randrange
from argparse import ArgumentParser
from copy import deepcopy

sys.path.append("../")
from gr.grdef import *
from gr.board import GrBoard
from gr.utils import get_image_area, resize

pattern = None
positive_dir = None
negative_dir = None
positive_dir_prefix = "p/"
negative_dir_prefix = "n/"
bw_sep = False
is_gen = False
n_free_space = 10
n_close_space = 1
neg_per_image = 0
p_method = "a"
n_resize = 0

def overlap(a, b):
    # from: https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
    x1 = max(min(a[0], a[2]), min(b[0], b[2]))
    y1 = max(min(a[1], a[3]), min(b[1], b[3]))
    x2 = min(max(a[0], a[2]), max(b[0], b[2]))
    y2 = min(max(a[1], a[3]), max(b[1], b[3]))
    return x1 < x2 and y1 < y2

def get_bg_color(img):
    # Find background color
    u, c =  np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    bg_c = u[c.argmax()]

    # Avoid selection of black-ish or white-ish colors
    if sum(bg_c) < 110 or sum(bg_c) >= 750:
        cc = c.argsort()
        n = -2
        while sum(bg_c) < 110 or sum(bg_c) >= 750:
            bg_c = u[cc[n]]
            n -= 1
    print('Background color is {}'.format(bg_c))
    return bg_c

def remove_areas(img, areas, bg_c):
    for c in areas:
        patch = np.full((c[3]-c[1], c[2]-c[0], img.shape[2]), bg_c, dtype = img.dtype)
        img[c[1]:c[3], c[0]:c[2]] = patch[:]
    return img

def get_space(r, space_str):
    n = str(space_str).find('%')
    if n == -1:
        return int(space_str)
    else:
        f = int(str(space_str)[0:n])
        return int(r * f / 100.0)

def get_free_space(r):
    return get_space(r, n_free_space)

def get_close_space(r):
    return get_space(r, n_close_space)


def one_file(file_name):
    # Open board
    print("Processing file " + str(file_name))
    try:
        board = GrBoard(str(file_name))
    except:
        print(sys.exc_info()[1])
        return

    # Init
    file_prefix = Path(file_name).stem + "_" + Path(file_name).suffix[1:]
    covered = []
    bg_c = get_bg_color(board.image)
    max_size = 0

    # Generate positive samples (board stones)
    pd = Path(positive_dir)
    if not bw_sep:
        f_reg = open(str(pd.joinpath("positives.txt")), "a")
    else:
        f_reg = {'B': open(str(pd.joinpath("b/positives.txt")), "a"),
                 'W': open(str(pd.joinpath("w/positives.txt")), "a")}

    ns = 0
    for stone in board.all_stones:
        x, y, a, b, r, bw = stone
        fs = get_free_space(r)
        cs = get_close_space(r)
        area_img = None
        area = None

        if p_method == "single" or p_method == 's':
            # Save single staying stones only
            nearby_stones = board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
            area = [max(x-r-fs,0),
                max(y-r-fs,0),
                min(x+r+fs, board.image.shape[CV_WIDTH]),
                min(y+r+fs, board.image.shape[CV_HEIGTH])]
            covered.extend([area])
            if len(nearby_stones) > 0: area = None

        elif p_method == "both" or p_method == 'b':
            # Saving all stones with different area square depending on
            # whether it has other stones nearby
            nearby_stones = board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
            if len(nearby_stones) == 0:
                area = [max(x-r-fs,0),
                    max(y-r-fs,0),
                    min(x+r+fs, board.image.shape[CV_WIDTH]),
                    min(y+r+fs, board.image.shape[CV_HEIGTH])]
            else:
                area = [max(x-r-cs,0),
                    max(y-r-cs,0),
                    min(x+r+cs, board.image.shape[CV_WIDTH]),
                    min(y+r+cs, board.image.shape[CV_HEIGTH])]
            covered.extend([area])

        elif p_method == "all" or p_method == 'a':
            # Save stone area with all stones except current one removed
            area = [max(x-r-fs,0),
                max(y-r-fs,0),
                min(x+r+fs, board.image.shape[CV_WIDTH]),
                min(y+r+fs, board.image.shape[CV_HEIGTH])]
            area_img = get_image_area(board.image, area)
            covered.extend([area])

            # Get stones nearby
            nearby_stones = board.stones.find_nearby((stone[GR_A], stone[GR_B]), 2)

            # Remove stones from the area
            nb_stone_areas = []
            for stone2 in nearby_stones:
                x2, y2, a2, b2, r2, bw2 = stone2
                nb_area = [min(max(x2-r2-area[0]-1,0), area_img.shape[CV_WIDTH]),
                    min(max(y2-r2-area[1]-1,0), area_img.shape[CV_HEIGTH]),
                    max(min(x2+r2-area[0], area_img.shape[CV_WIDTH]),0),
                    max(min(y2+r2-area[1], area_img.shape[CV_HEIGTH]),0)]
                if nb_area[0] < nb_area[2] and nb_area[1] < nb_area[3]:
                    nb_stone_areas.extend([nb_area])

            remove_areas(area_img, nb_stone_areas, bg_c)

        if area is not None:
            if area_img is None: area_img = get_image_area(board.image, area)
            covered.extend([area])

            fn = file_prefix + "_" + str(ns).zfill(3) + ".png"
            file_dir = pd.joinpath(bw.lower()) if bw_sep else pd
            save_fn = str(file_dir.joinpath(fn))
            print("\tPositive {} to {}".format(area, save_fn))

            if n_resize > 0:
                area_img = resize(area_img, n_resize, f_upsize = True, pad_color = bg_c)
            cv2.imwrite(save_fn, area_img)

            # Current sample creation implementation require relative path
            # from parent directory to be specified
            if not bw_sep:
                reg_fn = "{}{}".format(positive_dir_prefix, fn) if is_gen else fn
                f = f_reg
            else:
                reg_fn = "{}{}/{}".format(positive_dir_prefix, bw.lower(), fn) if is_gen else fn
                f = f_reg[bw]

            f.write("{} 1 {} {} {} {} \n".format(reg_fn,
                0, 0, area_img.shape[1]-1, area_img.shape[0]-1))
            ns += 1
            max_size = max([max_size, area_img.shape[1], area_img.shape[0]])

    if not bw_sep:
        f_reg.close()
    else:
        for k in f_reg: f_reg[k].close()

    # Generate negative samples (board areas without stones)
    pn = Path(negative_dir)

    # Prepare image with all found stones removed
    neg_img = remove_areas(board.image.copy(), covered, bg_c)
    fn = pn.joinpath(file_prefix + "_neg.png")
    cv2.imwrite(str(fn), neg_img)

    # Slice prepared image by random pieces generating number of
    # images not less than number of positive images
    w = min(max_size*2, int(round(neg_img.shape[CV_WIDTH] / 4,0)))
    h = min(max_size*2, int(round(neg_img.shape[CV_HEIGTH] / 4,0)))
    nn_max = neg_per_image if neg_per_image > 0 else ns

    f_reg = open(str(pn.joinpath("negatives.txt")), "a")
    for nn in range(nn_max):
        x = randrange(0, neg_img.shape[CV_WIDTH] - w)
        y = randrange(0, neg_img.shape[CV_HEIGTH] - h)

        area = [x, y, x + w, y + h]
        if area[0] < area[2] and area[1] < area[3]:
            im = get_image_area(neg_img, area)

            fn = pn.joinpath(file_prefix + "_" + str(nn).zfill(3) + ".png")
            print("\tNegative {} to {}".format(area, fn))

            cv2.imwrite(str(fn), im)
            f_reg.write("{}\n".format(negative_dir_prefix + str(fn.name)))

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


def main():
    global pattern, positive_dir_prefix, negative_dir_prefix

    # Ensure target directories exist and clean them up
    dir_list = [negative_dir, positive_dir]
    if bw_sep:
        dir_list.extend([Path(positive_dir).joinpath('b')])
        dir_list.extend([Path(positive_dir).joinpath('w')])

    for d in dir_list:
        pd = Path(d)
        pd.mkdir(exist_ok = True, parents = True)
        for x in pd.glob("*.*"):
            if x.is_file: os.remove(str(x))

    # Generate directory prefixes
    positive_dir_prefix = Path(positive_dir).stem + '/'
    negative_dir_prefix = Path(negative_dir).stem + '/'

    # Some info
    if p_method == "s":
        print("Saving single-staying stones only")
    elif p_method == "b":
        print("Saving all stones with different area size for single-staying/enclosed ones")
    elif p_method == "a":
        print("Saving all stones with surrounding area free of other stones")

    # Make pattern ready for glob:
    # Check it is a directory and if yes, add wildcards
    # If not, check for file wildcards, if none - add them
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.*")
    else:
        head, tail = os.path.split(pattern)
        print(head, " = ", tail)
        if tail == '': pattern = os.path.join(pattern, "*.*")

    # Iterate
    for x in glob.iglob(pattern):
        if os.path.isfile(x):
            y = Path(x)
            if y.suffix != '.gpar':
                # Image files processed as is
                one_file(y)
            else:
                # For .gpar files, try to find an image
                if y.with_suffix('.png').exists():
                    one_file(y.with_suffix('.png'))
                elif y.with_suffix('.jpg').exists():
                    one_file(y.with_suffix('.jpg'))
                else:
                    print("==> Cannot find an image which corresponds to {} param file".format(str(y)))

def get_args():
    global pattern, positive_dir, negative_dir, is_gen, n_free_space, n_close_space
    global bw_sep, neg_per_image, p_method, n_resize

    parser = ArgumentParser()
    parser.add_argument('pattern', help = 'Selection pattern')
    parser.add_argument('-p', "--positive",
        default = "./p/",
        help = 'Directory to store positive images (see also -b switch)')
    parser.add_argument('-n', "--negative",
        default = "./n/",
        help = 'Directory to store negative images')
    parser.add_argument('-b', "--bw",
        action = 'store_true',
        default = False,
        help = "If specified, black and white stone positive images ares stored separatelly in " +\
         "'b' and 'w' directories under specified positive directory")
    parser.add_argument('-m', "--method",
        choices = ["all", "both", "single"], default = "both",
        help = 'Image generation method, one of: ' + \
            "single - keep area of single-staying stones only, " + \
            "both - keep area of all stones but with different size for single or enclosed, " + \
            "all - keep areas of all stones with nearby stones removed"
        )
    parser.add_argument('-g', "--for_gen",
        action = "store_true",
        default = False,
        help = 'If specified, it is assumed that images will be used for samples generation')
    parser.add_argument('-f', "--space_free",
        default = 10,
        help = 'Space to add around single-staying stones, absolute number or percentage of stone radius followed by %')
    parser.add_argument('-c', "--space_close",
        default = 1,
        help = 'Space to add around stones having any nearby stone, absolute number or percentage of stone radius followed by %')
    parser.add_argument('-i', "--neg_img", type = int,
        default = 0,
        help = 'Number of negative images to generate from one image (0 - the same as positive, -1 - no negatives)')
    parser.add_argument('-r', "--resize", type = int,
        default = 0,
        help = 'Resize positive images to specified size')

    args = parser.parse_args()
    pattern = args.pattern
    positive_dir = args.positive
    negative_dir = args.negative
    bw_sep = args.bw
    p_method = args.method.lower()
    is_gen = args.for_gen
    n_free_space = args.space_free
    n_close_space = args.space_close
    neg_per_image = args.neg_img
    n_resize = args.resize


if __name__ == '__main__':
    get_args()
    main()
    cv2.destroyAllWindows()
