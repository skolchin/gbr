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
import os
import sys
import glob
import numpy as np
import cv2
import json
from pathlib import Path
from random import randrange
from argparse import ArgumentParser
from copy import deepcopy
from collections import OrderedDict

sys.path.append("../")
from gr.grdef import *
from gr.board import GrBoard
from gr.utils import get_image_area, resize, rotate

TF_AVAIL=True
try:
    import tensorflow as tf
    from object_detection.utils import dataset_util
    from tfrecord_lite import tf_record_iterator
except:
    TF_AVAIL=False

class DatasetRegistrator:
    """Supplementary class to handle description file creation"""

    def __init__(self, dir=None, reg_name='description.txt', subdir=''):
        self.dir, self.reg_name, self.subdir = dir, reg_name, subdir
        self.f_reg = None
        if self.dir is not None and self.reg_name is not None:
            self.open()

    def open(self, dir=None, reg_name='description.txt', subdir='', mode='a'):
        if dir is not None:
            self.dir = dir
        if reg_name is not None:
            self.reg_name = reg_name
        self.f_reg = open(str(Path(self.dir).joinpath(self.subdir, self.reg_name)), mode)

    def register_image(self, file_name, img):
        save_fn = str(Path(self.dir).joinpath(self.subdir, file_name))
        cv2.imwrite(save_fn, img)
        self.f_reg.write("{} 1 {} {} {} {} \n".format(
            file_name, 0, 0, img.shape[1]-1, img.shape[0]-1))

    def register_file(self, file_name):
        self.f_reg.write(str(file_name) + '\n')

    def close(self):
        if self.f_reg is not None:
            self.f_reg.close()
            self.f_reg = None

class DatasetGenerator:
    def __init__(self):
        # Datasets to generate
        self.datasets = ["positive", "negative", "stones", "crossings", 'bboxes']

        # Directories where to place datasets
        self.dirs = OrderedDict({"positive": None, "stones": None,
            "negative": None, "crossings": None, "bboxes": None})

        # Selection pattern
        self.pattern = None

        # Stone extraction method: single, enclosed, both
        self.method = "single"

        # Spacing of area to be extracted with particular method
        self.spacing = {"single": 10, "enclosed": 1, "crossing": 5, "bboxes": 1}

        # Number of negative areas to be extracted per image
        self.neg_per_image = 0

        # Resize maximum size
        self.n_resize = 0

        # Flag to exclude grid line crossings
        self.no_grid = False

        # Rotation vector (0: how many images to generate, 1: rotation angle)
        self.n_rotate = [0, 0]

        # BBox def file format (json, tf)
        bbox_fmt = 'json'

        # GrBoard currently processed
        self.board = None

        # Background color
        self.bg_c = None

        # Areas extracted during current run
        self.stone_areas = None

        # TF writer
        self.tf_writer = None

        # Statistic
        self.counts = {'positives': 0}
        self.totals = {'positives': 0}

    def overlap(self, a, b):
        """Check two rectangles overlap"""
        # from: https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
        x1 = max(min(a[0], a[2]), min(b[0], b[2]))
        y1 = max(min(a[1], a[3]), min(b[1], b[3]))
        x2 = min(max(a[0], a[2]), max(b[0], b[2]))
        y2 = min(max(a[1], a[3]), max(b[1], b[3]))
        return x1 < x2 and y1 < y2

    def get_bg_color(self, img):
        """Find background color of a board as most often occuring color except
            shades of black and white"""
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
        """Remove areas from image and pad it with background color"""
        for c in areas:
            patch = np.full((c[3]-c[1], c[2]-c[0], img.shape[2]), bg_c, dtype = img.dtype)
            img[c[1]:c[3], c[0]:c[2]] = patch[:]
        return img

    def get_image_file_name(self, src_file_name, index, ext = '.png'):
        """Derive image file from source file name"""
        prefix = Path(src_file_name).stem + "_" + Path(src_file_name).suffix[1:]
        return prefix + "_" + str(index).zfill(3) + ext

    def get_space(self, space, append_str):
        """Derive space to add to specfied integer space"""
        n = str(append_str).find('%')
        if n == -1:
            return int(append_str)
        else:
            append = int(str(append_str)[0:n])
            return int(space * append / 100.0)

    def get_registrator(self, ds_key, file_name='description.txt', subdir=''):
        """Create a dataset registrator for specified dataset"""
        return DatasetRegistrator(self.dirs[ds_key], file_name, subdir)

    def save_area(self, ds_key, file_name, area_img, start_index, f_reg, no_rotation=False):
        """Save given area of image file. If rotation is requested, generates it"""
        stop_index = start_index + 1 if self.n_rotate[0] == 0 or no_rotation \
                                     else start_index + self.n_rotate[0] + 1

        if self.n_resize > 0:
            area_img = resize(area_img, self.n_resize, f_upsize = True, pad_color = self.bg_c)

        bg_c = [int(x) for x in self.bg_c]
        for index in range(start_index, stop_index):
            fn = self.get_image_file_name(file_name, index)
            f_reg.register_image(fn, area_img)

            area_img = rotate(area_img, self.n_rotate[1], bg_c, keep_image=False)

        return stop_index - start_index

    def extract_stone_area(self, stone):
        x, y, a, b, r, bw = stone
        fs = self.get_space(r, self.spacing['single'])
        cs = self.get_space(r, self.spacing['enclosed'])
        area = None

        if self.method == "single" or self.method == 's':
            # Save single staying stones only
            nearby_stones = self.board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
            area = [max(x-r-fs,0),
                max(y-r-fs,0),
                min(x+r+fs, self.board.image.shape[CV_WIDTH]),
                min(y+r+fs, self.board.image.shape[CV_HEIGTH])]
            self.stone_areas.extend([area])
            if len(nearby_stones) > 0: area = None

        if self.method == "enclosed" or self.method == 'e':
            # Save enclosed staying stones only
            nearby_stones = self.board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
            area = [max(x-r-cs,0),
                max(y-r-cs,0),
                min(x+r+cs, self.board.image.shape[CV_WIDTH]),
                min(y+r+cs, self.board.image.shape[CV_HEIGTH])]
            self.stone_areas.extend([area])
            if len(nearby_stones) == 0: area = None

        elif self.method == "both" or self.method == 'b':
            # Saving all stones with different area square depending on
            # whether it has other stones nearby
            nearby_stones = self.board.stones.find_nearby((stone[GR_A], stone[GR_B]), 1)
            if len(nearby_stones) == 0:
                area = [max(x-r-fs,0),
                    max(y-r-fs,0),
                    min(x+r+fs, self.board.image.shape[CV_WIDTH]),
                    min(y+r+fs, self.board.image.shape[CV_HEIGTH])]
            else:
                area = [max(x-r-cs,0),
                    max(y-r-cs,0),
                    min(x+r+cs, self.board.image.shape[CV_WIDTH]),
                    min(y+r+cs, self.board.image.shape[CV_HEIGTH])]
            self.stone_areas.extend([area])
        return area

    def extract_crossing_range(self, file_name, subdir, ranges):
        f_reg = self.get_registrator('crossings', subdir=subdir)
        cs = self.get_space(4, self.spacing['crossing'])
        index = 0

        for r in ranges:
            for y in r[0]:
                for x in r[1]:
                    stone = self.board.find_stone(c=(x,y))
                    if stone is None:
                        area = [max(x-cs-2,0),
                            max(y-cs-2,0),
                            min(x+cs+2, self.board.image.shape[CV_WIDTH]),
                            min(y+cs+2, self.board.image.shape[CV_HEIGTH])]

                        area_img = get_image_area(self.board.image, area)

                        n = self.save_area('crossings', file_name, area_img,
                                            index, f_reg)
                        index += n
                        self.counts['crossings'] += n

        f_reg.close()


    def extract_border_crossings(self, file_name):
        edges = self.board.results[GR_EDGES]
        space = self.board.results[GR_SPACING]

        ranges = [
            # left border
            (
                range(int(edges[0][1]+space[1]), int(edges[1][1]-space[1]), int(space[1])),
                range(int(edges[0][0]), int(edges[0][0])+1, int(space[0]))
            ),
            # right border
            (
                range(int(edges[0][1]+space[1]), int(edges[1][1]-space[1]), int(space[1])),
                range(int(edges[1][0]), int(edges[1][0])+1, int(space[0]))
            ),
            # top border
            (
                range(int(edges[0][1]), int(edges[0][1])+1, int(space[1])),
                range(int(edges[0][0]+space[0]), int(edges[1][0]-space[0]), int(space[0]))
            ),
            # bottom border
            (
                range(int(edges[1][1]), int(edges[1][1])+1, int(space[1])),
                range(int(edges[0][0]+space[0]), int(edges[1][0]-space[0]), int(space[0]))
            )
        ]
        self.extract_crossing_range(file_name, 'border', ranges)


    def extract_inboard_crossings(self, file_name):
        edges = self.board.results[GR_EDGES]
        space = self.board.results[GR_SPACING]

        ranges = [
            (
                range(int(edges[0][1]+space[1]), int(edges[1][1]-space[1]), int(space[1])),
                range(int(edges[0][0]+space[0]), int(edges[1][0]-space[0]), int(space[0]))
            )
        ]
        self.extract_crossing_range(file_name, "cross", ranges)

    def extract_edges(self, file_name):
        edges = self.board.results[GR_EDGES]
        ranges = [
            (
                [edges[0][1]],
                [edges[0][0]]
            ),
            (
                [edges[1][1]],
                [edges[0][0]]
            ),
            (
                [edges[0][1]],
                [edges[1][0]]
            ),
            (
                [edges[1][1]],
                [edges[1][0]]
            ),
        ]
        self.extract_crossing_range(file_name, 'edge', ranges)


    def extract_positive(self, file_name):
        """Positives (stones) dataset extractor"""
        f_reg = self.get_registrator('positive', 'positives.txt')
        index = 0

        for stone in self.board.all_stones:
            area = self.extract_stone_area(stone)
            if area is not None:
                area_img = get_image_area(self.board.image, area)
                n = self.save_area('positive', file_name, area_img, index, f_reg)
                index += n
                self.counts['positive'] += n

        f_reg.close()

    def extract_negative(self, file_name):
        """Negatives (empty boards) dataset extractor"""
        # Prepare image with all found stones removed
        neg_img = self.remove_areas(self.board.image.copy(), self.stone_areas, self.bg_c)
        fn = self.get_image_file_name(file_name, 999).replace('999', 'neg')
        self.save_image('negative', fn, neg_img)

        # Slice prepared image by random pieces generating number of
        # images not less than specified number
        w = int(round(neg_img.shape[CV_WIDTH] / 4,0))
        h = int(round(neg_img.shape[CV_HEIGTH] / 4,0))
        nn_max = self.neg_per_image if self.neg_per_image > 0 else self.counts['positive']

        f_reg = self.get_registrator('negative', 'negatives.txt')
        for index in range(nn_max):
            x = randrange(0, neg_img.shape[CV_WIDTH] - w)
            y = randrange(0, neg_img.shape[CV_HEIGTH] - h)

            area = [x, y, x + w, y + h]
            if area[0] < area[2] and area[1] < area[3]:
                area_img = get_image_area(neg_img, area)
                n = self.save_area('negative', file_name, area_img, index, f_reg)
                self.counts['negative'] += n

        f_reg.close()

    def extract_stones(self, file_name):
        """Stones dataset extractor"""

        def extract_stones_bw(bw, subdir):
            f_reg = self.get_registrator('stones', subdir = subdir)
            index = 0
            stones = [x for x in self.board.all_stones if x[GR_BW] == bw]

            for stone in stones:
                area = self.extract_stone_area(stone)
                if area is not None:
                    area_img = get_image_area(self.board.image, area)
                    n = self.save_area('stones', file_name, area_img, index, f_reg)
                    self.counts['stones'] += n

            f_reg.close()

        extract_stones_bw('B', 'black')
        extract_stones_bw('W', 'white')

    def extract_crossings(self, file_name):
        """Crossings dataset extractor"""
        self.extract_edges(file_name)
        self.extract_border_crossings(file_name)
        if not self.no_grid:
            self.extract_inboard_crossings(file_name)

    def extract_bboxes(self, file_name):
        """Extractor which creates whole board description in TF-REC format"""

        # Get registrator
        f_reg = self.get_registrator('bboxes')

        # Generate label map
        fn_map = Path(self.dirs['bboxes']).joinpath('go_board.names')
        if not fn_map.exists():
            with open(str(fn_map), 'w') as f_map:
                f_map.write('item {\n' + \
                            '  id: 1\n' + \
                            '  name: \'black\'\n' + \
                            '}\n' + \

                            'item {\n' + \
                            '  id: 2\n' + \
                            '  name: \'white\'\n' + \
                            '}\n')

            f_map.close()

        # Generate features
        if self.bbox_fmt == 'json':
            # JSON format
            height, width = self.board.image.shape[:2]
            features = {
                'image/height': height,
                'image/width': width,
                'image/filename': str(Path(file_name).parts[-1]),
                'image/source_id': str(Path(file_name).parts[-1]),
                'image/encoded': None,
                'image/format': str(Path(file_name).suffix[1:]),
                'image/objects': []
            }
            for stone in self.board.all_stones:
                area = self.extract_stone_area(stone)
                if area is not None:
                    txt, lbl = ('black', 1) if stone[5] == 'B' else ('white', 2)
                    features['image/objects'].append({'bbox': area, 'class/text': txt, 'class/label': lbl})

            fn_bbox = self.get_image_file_name(file_name, 1, ext='.json')
            with open(str(Path(self.dirs['bboxes']).joinpath(fn_bbox)), 'w') as f_bbox:
                json.dump(features, f_bbox, ensure_ascii=False, indent=4)
                self.counts['bboxes'] += 1
                f_bbox.close()

        elif self.bbox_fmt == 'tf':
            # TF-record format
            height, width = self.board.image.shape[:2]
            txts, lbls, xmins, xmaxs, ymins, ymaxs = [], [], [], [], [], []
            for stone in self.board.all_stones:
                area = self.extract_stone_area(stone)
                if area is not None:
                    txt, lbl = ('black', 1) if stone[5] == 'B' else ('white', 2)
                    txts.append(str.encode(txt))
                    lbls.append(lbl)
                    xmins.append(area[0] / width)
                    ymins.append(area[1] / height)
                    xmaxs.append(area[2] / width)
                    ymaxs.append(area[3] / height)

            fname, fmt = str(Path(file_name).parts[-1]), str(Path(file_name).suffix[1:])
            features = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(str.encode(fname)),
                'image/source_id': dataset_util.bytes_feature(str.encode(fname)),
                'image/encoded': dataset_util.bytes_feature(b''),
                'image/format': dataset_util.bytes_feature(str.encode(fmt)),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(txts),
                'image/object/class/label': dataset_util.int64_list_feature(lbls)
            }))
            self.tf_writer.write(features.SerializeToString())
            self.counts['bboxes'] += 1

        else:
            raise ValueError('Invalid bbox format or TF is not available')

        f_reg.register_file(file_name)
        f_reg.close()

    def one_file(self, file_name):
        # Open board
        print("Processing file " + str(file_name))
        try:
            self.board = GrBoard(str(file_name))
        except:
            print(sys.exc_info()[1])
            return

        self.bg_c = self.get_bg_color(self.board.image)
        self.stone_areas = []
        for k in self.counts: self.counts[k] = 0

        for k in self.datasets:
            extractor_fn = getattr(self, 'extract_' + k, None)
            if extractor_fn is None:
                raise ValueError('Cannot find a handler to generate dataset ', k)
            else:
                extractor_fn(file_name)

        for k in self.counts: self.totals[k] += self.counts[k]


    def get_args(self):
        parser = ArgumentParser()
        parser.add_argument('pattern', help = 'Selection pattern')
        parser.add_argument('-p', '--positive',
            help = 'Directory to store positives dataset (images with stones)')
        parser.add_argument('-n', '--negative',
            help = 'Directory to store negatives dataset (images without stones)')
        parser.add_argument('-s', '--stones',
            help = "Directory to store stones dataset (stone images, separately for black and white)")
        parser.add_argument('-c', '--crossings',
            help = "Directory to store line crossings and edges dataset (images of board grid lines crossings, " + \
                    "separately for edges, borders crossings and grid lines crossings)")
        parser.add_argument('-b', '--bboxes',
            help = "Directory to store bboxes dataset")
        parser.add_argument('-m', '--method',
            choices = ["single", "enclosed", "both"], default = "both",
            help = "Stone image extration method, one of: " + \
                "single - extract areas of single-staying stones, " + \
                "enclosed - extract areas of stones enclosed by other stones, " + \
                "both - extract all stones")
        parser.add_argument('--space',
            nargs = '*',
            default = [10, 3, 5],
            help = "Space to add when extracting area for: single stones, " + \
                    "enclosed stones, edges/crossings " + \
                    "(numbers or perecentage of stone size followed by %)")
        parser.add_argument('--neg-img', type = int,
            default = 0,
            help = 'Number of negative images to generate from one image (0 - the same number as positives)')
        parser.add_argument('--resize', type = int,
            default = 0,
            help = 'Resize images to specified size (0 - no resizing)')
        parser.add_argument('--no-grid',
            action="store_true",
            default = False,
            help = 'Do not generate grid line crossing images')
        parser.add_argument('--rotate', type = int,
            nargs = 2,
            default = [0, 0],
            help = 'Two numbers specifying how many rotation images shall be created and an angle for each rotation')
        parser.add_argument('--bbox-fmt',
            choices = ["json", "tf"], default = "json",
            help = "BBoxes definition format, json or tf-record (if Tensorflow is available)")

        args = parser.parse_args()
        self.dirs['positive'] = args.positive
        self.dirs['stones'] = args.stones
        self.dirs['negative'] = args.negative
        self.dirs['crossings'] = args.crossings
        self.dirs['bboxes'] = args.bboxes

        self.datasets = [x for x in self.dirs if self.dirs[x] is not None]
        if len(self.datasets) == 0:
            raise ValueError('No datasets to generate')

        self.pattern = args.pattern
        self.method = args.method.lower()

        self.spacing['single'] = args.space[0]
        self.spacing['enclosed'] = args.space[1] if len(args.space) > 1 else 1
        self.spacing['crossing'] = args.space[2] if len(args.space) > 2 else 5

        self.neg_per_image = args.neg_img
        self.n_resize = args.resize
        self.no_grid = args.no_grid
        self.n_rotate = args.rotate
        self.bbox_fmt= args.bbox_fmt
        if self.bbox_fmt == 'tf' and not TF_AVAIL:
            print('TF-record bboxes output requested, but TF is not available. Switching to JSON')
            self.bbox_fmt = 'json'

    def main(self):
        self.get_args()

        # Ensure target directories exist and clean them up
        dir_list = [x for x in self.dirs.values() if x is not None]
        if self.dirs.get('stones') is not None:
            dir_list.extend([Path(self.dirs['stones']).joinpath('black')])
            dir_list.extend([Path(self.dirs['stones']).joinpath('white')])

        if self.dirs.get('crossings') is not None:
            dir_list.extend([Path(self.dirs['crossings']).joinpath('edge')])
            dir_list.extend([Path(self.dirs['crossings']).joinpath('border')])
            if not self.no_grid:
                dir_list.extend([Path(self.dirs['crossings']).joinpath('cross')])

        for d in dir_list:
            pd = Path(d)
            pd.mkdir(exist_ok = True, parents = True)
            for x in pd.glob("*.*"):
                if x.is_file: os.remove(str(x))

        # Make pattern ready for glob:
        # Check it is a directory and if yes, add wildcards
        # If not, check for file wildcards, if none - add them
        if os.path.isdir(self.pattern):
            self.pattern = os.path.join(self.pattern, "*.*")
        else:
            head, tail = os.path.split(self.pattern)
            if tail == '': pattern = os.path.join(self.pattern, "*.*")

        # Prepare stats
        self.totals = {k: 0 for k in self.datasets}
        self.counts = self.totals.copy()

        # Get a TF writer if neeeded
        if self.bbox_fmt == 'tf' and self.tf_writer is None:
            self.tf_writer = \
                tf.io.TFRecordWriter(str(Path(self.dirs['bboxes']).joinpath('go_board.tfrecord')))

        # Iterate
        for x in glob.iglob(self.pattern):
            if os.path.isfile(x):
                if Path(x).suffix != '.gpar':
                    # Image files processed as is
                    self.one_file(x)
                else:
                    # For .gpar files, try to find an image
                    found = False
                    for sx in ['.png', '.jpg', '.jpeg']:
                        f = Path(x).with_suffix(sx)
                        found = f.exists()
                        if found:
                            self.one_file(f)
                            break

                    if not found:
                        print("==> Cannot find an image which corresponds to {} param file".format(x))

        # Statistics
        print("Files generated:")
        for k, v in self.totals.items():
            print("\t{}: {}".format(k, v))

        # Clean up
        if self.tf_writer is not None:
            self.tf_writer.close()

if __name__ == '__main__':
    app = DatasetGenerator()
    app.main()
    cv2.destroyAllWindows()

##    it = tf_record_iterator("C:\\Users\\kol\\Documents\\kol\\gbr\\cc\\bboxes\\go_board.tfrecord")
##    for n, r in enumerate(it):
##        print('==> ' + str(n))
##        print(r)
##        print('')
