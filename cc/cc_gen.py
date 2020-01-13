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

class DatasetGenerator:
    def __init__(self):
        # Datasets to generate: positives, negatives, bw, crosses, edges
        self.datasets = ["positives"]

        # Directories where to place datasets
        self.dirs = {"positives": "./cc/p/"}

        # Selection pattern
        self.pattern = None

        # Stone extraction method: single, enclosed, both
        self.method = "single"

        # Spacing of area to be extracted with particular method
        self.spacing = {"single": 10, "enclosed": 1}

        # Positive dataset is for samples generation
        self.is_gen = False

        # Number of negative areas to be extracted per image
        self.neg_per_image = 0

        # Resize
        self.n_resize = 0

        # Processing results
        self.counts = {'positives': 0}
        self.totals = {'positives': 0}

    def overlap(self, a, b):
        # from: https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
        x1 = max(min(a[0], a[2]), min(b[0], b[2]))
        y1 = max(min(a[1], a[3]), min(b[1], b[3]))
        x2 = min(max(a[0], a[2]), max(b[0], b[2]))
        y2 = min(max(a[1], a[3]), max(b[1], b[3]))
        return x1 < x2 and y1 < y2

    def get_bg_color(self, img):
        # Find background color
        u, c =  np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        bg_c = u[c.argmax()]

        # Check black or white color selected
        if sum(bg_c) < 40 or sum(bg_c) >= 750:
            cc = c.argsort()
            n = -2
            while sum(bg_c) < 40 or sum(bg_c) >= 750:
                bg_c = u[cc[n]]
                n -= 1
        return bg_c

    def remove_areas(self, img, areas, bg_c):
        for c in areas:
            patch = np.full((c[3]-c[1], c[2]-c[0], img.shape[2]), bg_c, dtype = img.dtype)
            img[c[1]:c[3], c[0]:c[2]] = patch[:]
        return img

    def get_image_file_name(self, src_file_name, index, ext = '.png'):
        prefix = Path(src_file_name).stem + "_" + Path(src_file_name).suffix[1:]
        return prefix + "_" + str(index).zfill(3) + ext

    def extract_positives(self, file_name, board):
        pass

    def extract_negatives(self, file_name, board):
        pass

    def extract_stones(self, file_name, board):
        pass

    def extract_crosses(self, file_name, board):
        pass

    def extract_edges(self, file_name, board):
        pass

    def one_file(self, file_name):
        # Open board
        print("Processing file " + str(file_name))
        board = GrBoard(str(file_name))
        prefix = Path(file_name).stem + "_" + Path(file_name).suffix[1:]
        covered = []
        bg_c = self.get_bg_color(board.image)
        max_size = 0

        # Generate positive samples (board stones)
        pd = Path(positive_dir)
        f_reg = open(str(pd.joinpath("positives.txt")), "a")
        ns = 0
        for stone in board.all_stones:
            x, y, a, b, r, bw = stone
            area_img = None
            area = None

            if p_method == "s":
                # Save single staying stones only
                nearby_stones = board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
                area = [max(x-r-free_space,0),
                    max(y-r-free_space,0),
                    min(x+r+free_space, board.image.shape[CV_WIDTH]),
                    min(y+r+free_space, board.image.shape[CV_HEIGTH])]
                covered.extend([area])
                if len(nearby_stones) > 0: area = None

            elif p_method == "b":
                # Saving all stones with different area square
                nearby_stones = board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
                if len(nearby_stones) == 0:
                    area = [max(x-r-free_space,0),
                        max(y-r-free_space,0),
                        min(x+r+free_space, board.image.shape[CV_WIDTH]),
                        min(y+r+free_space, board.image.shape[CV_HEIGTH])]
                else:
                    area = [max(x-r-close_space,0),
                        max(y-r-close_space,0),
                        min(x+r+close_space, board.image.shape[CV_WIDTH]),
                        min(y+r+close_space, board.image.shape[CV_HEIGTH])]
                covered.extend([area])

            elif p_method == "a":
                # Save large areas with all stones except current one removed
                area = [max(x-r-free_space,0),
                    max(y-r-free_space,0),
                    min(x+r+free_space, board.image.shape[CV_WIDTH]),
                    min(y+r+free_space, board.image.shape[CV_HEIGTH])]
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

                fn = pd.joinpath(prefix + "_" + str(ns).zfill(3) + ".png")
                print("\tPositive {} to {}".format(area, str(fn)))

                if n_resize > 0:
                    area_img = resize(area_img, n_resize, f_upsize = True, pad_color = bg_c)
                cv2.imwrite(str(fn), area_img)

                p = "p/" if is_gen else ""
                f_reg.write("{} 1 {} {} {} {} \n".format(p + str(fn.name),
                    0, 0, area_img.shape[1]-1, area_img.shape[0]-1))
                ns += 1
                max_size = max([max_size, area_img.shape[1], area_img.shape[0]])

        f_reg.close()

        # Generate negative samples (board areas without stones)
        pn = Path(negative_dir)

        # Prepare board image with stones removed
        neg_img = remove_areas(board.image.copy(), covered, bg_c)
        fn = pn.joinpath(prefix + "_neg.png")
        cv2.imwrite(str(fn), neg_img)

        # Slice the no-stones image by random pieces generating number of
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

                fn = pn.joinpath(prefix + "_" + str(nn).zfill(3) + ".png")
                print("\tNegative {} to {}".format(area, fn))

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


    def main():
        global pattern

        pd = Path(positive_dir)
        pd.mkdir(exist_ok = True, parents = True)
        for x in pd.glob("*.*"):
            if x.is_file: os.remove(str(x))

        pn = Path(negative_dir)
        pn.mkdir(exist_ok = True, parents = True)
        for x in pn.glob("*.*"):
            if x.is_file: os.remove(str(x))

        if p_method == "s":
            print("Saving single-staying stones only")
        elif p_method == "b":
            print("Saving all stones with different area for single/closed ones")
        elif p_method == "a":
            print("Saving all stones with surrounding area free of other stones")

        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*.*")
        else:
            head, tail = os.path.split(pattern)
            print(head, " = ", tail)
            if tail == '': pattern = os.path.join(pattern, "*.*")

        for x in glob.iglob(pattern):
            if os.path.isfile(x):
                y = Path(x)
                if y.suffix != '.gpar':
                    one_file(y)
                else:
                    if y.with_suffix('.png').exists():
                        one_file(y.with_suffix('.png'))
                    elif y.with_suffix('.jpg').exists():
                        one_file(y.with_suffix('.jpg'))


    def get_args(self):
        parser = ArgumentParser()
        parser.add_argument('pattern', help = 'Selection pattern')
        parser.add_argument('-p', "--positive",
            help = 'Directory to store positive dataset')
        parser.add_argument('-n', "--negative",
            help = 'Directory to store negative dataset')
        parser.add_argument('-b', "--bw",
            help = 'Directory to store stones dataset (\'black\' and \'white\' subdirectories will be created')

        parser.add_argument('-m', "--method",
            choices = ["single", "enclosed", "both"], default = "all",
            help = 'Positive or stone extraction method, one of: ' + \
                "single - extract areas around single-staying stones, " + \
                "enclosed - extract areas around stones enclosed by other stones" + \
                "a - keep areas of all stones with nearby stones removed"
            )
        parser.add_argument('-g', "--for_gen", type = int,
            choices = [0, 1], default = 0,
            help = 'Set to 1 if images will be used for samples generation, 0 otherwise')
        parser.add_argument('-f', "--space_free", type = int,
            default = 10,
            help = 'Space to add around free-staying stones')
        parser.add_argument('-c', "--space_close", type = int,
            default = 1,
            help = 'Space to add around stones having any nearby stone')
        parser.add_argument('-i', "--neg_img", type = int,
            default = 0,
            help = 'Number of negative images to generate from one image (0 - the same as positive)')
        parser.add_argument('-r', "--resize", type = int,
            default = 0,
            help = 'Resize positive image to specified size')

        args = parser.parse_args()
        pattern = args.pattern
        positive_dir = args.positive
        negative_dir = args.negative
        p_method = args.method
        is_gen = args.for_gen > 0
        free_space = args.space_free
        close_space = args.space_close
        neg_per_image = args.neg_img
        n_resize = args.resize


if __name__ == '__main__':
    get_args()
    main()
    cv2.destroyAllWindows()
