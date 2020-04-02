#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     OpenCV board extraction script
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
from random import randint, randrange, shuffle
from argparse import ArgumentParser
from copy import deepcopy
from collections import OrderedDict

#sys.path.append("../")
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


class DatasetWriter:
    """Basic dataset writer"""
    def __init__(self, root_dir, file_count, split, reg_name):
        self.root_dir, self.file_count = root_dir, file_count
        self.reg_name, self.split = reg_name, split

        self.file_index, self.file_name, self.file_image = None, None, None
        self.area_index, self.area_count = None, None

    def get_next_fname(self, label=None, ext='.png'):
        """Derive image file from source file name"""
        prefix = Path(self.file_name).stem + "_" + Path(self.file_name).suffix[1:]

        fn = str(prefix) + "_" + str(self.area_index).zfill(3) + ext
        path = []

        if label is not None:
            path.append(label)
        if self.split is not None and self.split > 0:
            if self.area_count is None:
                raise ValueError('Area count must be set')

            if self.area_index < self.area_count * (1.0 - self.split):
                path.append('train')
            else:
                path.append('val')

        path.append(fn)
        fn = Path(self.root_dir).joinpath(*path)
        self.area_index += 1

        return fn

    def get_reg_fname(self, label=None):
        """Get registration file name"""
        path = []

        if label is not None:
            path.append(label)
        if self.split is not None and self.split > 0:
            if self.area_count is None:
                raise ValueError('Area count must be set')

            if self.area_index < self.area_count * (1.0 - self.split):
                path.append('train')
            else:
                path.append('val')

        path.append(self.reg_name)
        fn = Path(self.root_dir).joinpath(*path)
        fn.parent.mkdir(exist_ok=True, parents=True)
        return fn

    def set_image(self, file_index, file_name, file_image, area_count):
        self.file_index, self.file_name, self.file_image = file_index, file_name, file_image
        self.area_index, self.area_count = 0, area_count

        self.write_image_info()

    def write_image_info(self):
        pass

    def write_area(self, area, label=None):
        area_img = get_image_area(self.file_image, area)
        if area_img is not None:
            self.write_area_image(area, area_img, label)

    def write_area_image(self, area, area_image, label=None):
        pass

    def finish_image(self):
        pass

    def close(self):
        pass

class TxtDatasetWriter(DatasetWriter):
    def __init__(self, root_dir, file_count, split=None, reg_name='description.txt'):
        super(TxtDatasetWriter, self).__init__(root_dir, file_count, split, reg_name)

    def write_area_image(self, area, area_image, label=None):
        # Define file names
        mode = 'w' if self.file_index == 0 else 'a'
        fn_area = self.get_next_fname(label)
        fn_reg = self.get_reg_fname(label)

        # Save file
        cv2.imwrite(str(fn_area), area_image)

        # Save file info
        with open(str(fn_reg), mode) as f_reg:
            f_reg.write('{} 1 {} {} {} {}\n'.format(
                fn_area, 0, 0, area_image.shape[1]-1, area_image.shape[0]-1))
            f_reg.close()


class TfDatasetWriter(DatasetWriter):
    def __init__(self, root_dir, file_count, split=None, reg_name='description.txt',
                 base_name='go_board.tfrecord'):
        super(TfDatasetWriter, self).__init__(root_dir, file_count, split, reg_name)

        self.tf_writers = {'all': None, 'train': None, 'val': None}
        self.txts, self.lbls, self.xmins, self.xmaxs, self.ymins, self.ymaxs = [], [], [], [], [], []

        # Writer mode, either `single` (whole board image file) or `multi` (multiple image parts)
        self.mode = 'single'

        # Get TF writers
        if self.split is None:
            self.tf_writers['all'] = tf.io.TFRecordWriter(str(Path(self.root_dir).joinpath(base_name)))
        else:
            for k in ['train', 'val']:
                fn = Path(base_name).stem + '_' + k + Path(base_name).suffix
                self.tf_writers[k] = tf.io.TFRecordWriter(str(Path(self.root_dir).joinpath(fn)))

    def write_image_info(self):
        self.txts, self.lbls, self.xmins, self.xmaxs, self.ymins, self.ymaxs = [], [], [], [], [], []

    def write_area(self, area, label):
        self.mode = 'single'
        height, width = self.file_image.shape[:2]
        self.txts.append(str.encode(label, 'utf-8'))
        self.lbls.append(1 if label == 'black' else 2)
        self.xmins.append(area[0] / width)
        self.ymins.append(area[1] / height)
        self.xmaxs.append(area[2] / width)
        self.ymaxs.append(area[3] / height)

    def write_area_image(self, area, area_image, label):
        self.mode = 'multi'
        height, width = area_image.shape[:2]
        txts = [str.encode(label, 'utf-8')]
        lbls = [1 if label == 'black' else 2]
        xmins = [area[0] / width]
        ymins = [area[1] / height]
        xmaxs = [area[2] / width]
        ymaxs = [area[3] / height]
        fn_area = self.get_next_fname(label)

        self.save_image(
            self.area_index, fn_area, area_image, self.area_count,
            [txts, lbls, xmins, xmaxs, ymins, ymaxs]
        )

    def finish_image(self):
        if self.mode == 'single':
            self.save_image(
                self.file_index, self.file_name, self.file_image, self.file_count,
                [self.txts, self.lbls, self.xmins, self.xmaxs, self.ymins, self.ymaxs]
            )

    def encode_image(self, image):
        # Encode image as JPEG via temp file
        fn_tmp = Path(self.root_dir).joinpath('_tmp_' + str(randint(1,100)) + '.jpeg')
        cv2.imwrite(str(fn_tmp), image)
        with open(str(fn_tmp), 'rb') as f_tmp:
            img_raw = f_tmp.read()
            f_tmp.close()
        fn_tmp.unlink()
        return img_raw

    def save_image(self, image_index, image_file, image, image_count, feature_comps):
        # Get encoded image
        img_raw = self.encode_image(image)

        # Make up features
        height, width = image.shape[:2]
        fname = str(Path(image_file).parts[-1])
        txts, lbls, xmins, xmaxs, ymins, ymaxs = feature_comps

        features = tf.train.Example(features=tf.train.Features(feature={
            'image/width': dataset_util.int64_feature(width),
            'image/height': dataset_util.int64_feature(height),
            'image/filename': dataset_util.bytes_feature(str.encode(fname, 'utf-8')),
            'image/source_id': dataset_util.bytes_feature(str.encode(fname, 'utf-8')),
            'image/format': dataset_util.bytes_feature(b'jpg'),
            'image/encoded': dataset_util.bytes_feature(img_raw),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(txts),
            'image/object/class/label': dataset_util.int64_list_feature(lbls)
        }))

        # Save to appropriate TF writer
        tf_writer = self.tf_writers['all']
        if tf_writer is None:
            tf_writer = self.tf_writers['train'] \
                        if image_index < image_count * (1.0 - self.split) \
                        else self.tf_writers['val']

        tf_writer.write(features.SerializeToString())

        # Register file
        mode = 'w' if image_index == 0 else 'a'
        fn_reg = str(Path(self.root_dir).joinpath(self.reg_name))
        with open(str(fn_reg), mode) as f_reg:
            f_reg.write('{}\n'.format(image_file))
            f_reg.close()

    def close(self):
        for w in self.tf_writers.values():
            if w is not None: w.close()

class DatasetGenerator:
    """ Main dataset generator class"""

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

        # Dataset output format (txt, json, xml, tf)
        self.format = 'txt'

        # Dataset split and shuffle flags
        self.split = None
        self.shuffle = False

        # GrBoard currently processed
        self.board = None

        # Background color
        self.bg_c = None

        # Areas extracted during current run
        self.stone_areas = None

        # Dataset writers
        self.ds_writers = None

        # Statistic
        self.file_count = 0
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

    def get_space(self, space, append_str):
        """Derive space to add to specfied integer space"""
        n = str(append_str).find('%')
        if n == -1:
            return int(append_str)
        else:
            append = int(str(append_str)[0:n])
            return int(space * append / 100.0)

    def save_area(self, ds_key, file_name, area_img, start_index, f_reg, no_rotation=False):
        """Save given area of image file. If rotation is requested, generates it"""
        stop_index = start_index + 1 if self.n_rotate[0] == 0 or no_rotation \
                                     else start_index + self.n_rotate[0] + 1

        if min(self.n_resize) > 0:
            area_img = resize(area_img, self.n_resize, f_upsize=True, pad_color=self.bg_c)

        bg_c = [int(x) for x in self.bg_c]
        for index in range(start_index, stop_index):
            fn = self.get_image_file_name(file_name, index)
            f_reg.register_image(fn, area_img)

            area_img = rotate(area_img, self.n_rotate[1], bg_c, keep_image=False)

        return stop_index - start_index

    def add_count(self, key):
        if key in self.counts:
            self.counts[key] += 1
        else:
            self.counts[key] = 1

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

    def extract_crossing_range(self, file_index, file_name, label, ranges):
        # Get all crossings in a list
        cs = self.get_space(4, self.spacing['crossing'])
        crossings = []
        for r in ranges:
            for y in r[0]:
                for x in r[1]:
                    stone = self.board.find_stone(c=(x,y))
                    if stone is None:
                        area = [max(x-cs-2,0),
                            max(y-cs-2,0),
                            min(x+cs+2, self.board.image.shape[CV_WIDTH]),
                            min(y+cs+2, self.board.image.shape[CV_HEIGTH])]

                        crossings.append(area)

        # Prepare the writer
        self.ds_writers['crossings'].set_image(file_index,
                                               file_name,
                                               self.board.image,
                                               len(crossings))

        # Proceed
        for area in crossings:
            area_img = get_image_area(self.board.image, area)
            self.ds_writers['crossings'].write_area_image(area, area_img, label)
            self.add_count('crossings\\' + label)

        # Finalize
        self.ds_writers['crossings'].finish_image()

    def extract_border_crossings(self, file_index, file_name):
        """External grid crossings dataset extractor. Called from within crossings extractor"""
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
        self.extract_crossing_range(file_index, file_name, 'border', ranges)


    def extract_inboard_crossings(self, file_index, file_name):
        """Internal grid crossing dataset extractor. Called from within crossings extractor"""
        edges = self.board.results[GR_EDGES]
        space = self.board.results[GR_SPACING]

        ranges = [
            (
                range(int(edges[0][1]+space[1]), int(edges[1][1]-space[1]), int(space[1])),
                range(int(edges[0][0]+space[0]), int(edges[1][0]-space[0]), int(space[0]))
            )
        ]
        self.extract_crossing_range(file_index, file_name, "cross", ranges)

    def extract_edges(self, file_index, file_name):
        """Edges dataset extractor. Called from within crossings extractor"""
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
        self.extract_crossing_range(file_index, file_name, 'edge', ranges)

    def extract_positive(self, file_index, file_name):
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

    def extract_negative(self, file_index, file_name):
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

    def extract_stones(self, file_index, file_name):
        """Stones dataset extractor"""
        # Prepare the writer
        self.ds_writers['stones'].set_image(file_index,
                                            file_name,
                                            self.board.image,
                                            len(self.board.all_stones))


        # Shuffle stones list if split is requested
        stones = deepcopy(self.board.all_stones)
        if self.shuffle and self.split > 0:
            shuffle(stones)

        # Process stones
        for stone in stones:
            label = 'black' if stone[GR_BW] == 'B' else 'white'
            area = self.extract_stone_area(stone)
            if area is not None:
                area_img = get_image_area(self.board.image, area)
                self.ds_writers['stones'].write_area_image(area, area_img, label)
                self.add_count('stones\\' + label)

        # Finalize
        self.ds_writers['stones'].finish_image()

    def extract_crossings(self, file_index, file_name):
        """Crossings dataset extractor"""
        self.extract_edges(file_index, file_name)
        self.extract_border_crossings(file_index, file_name)
        if not self.no_grid:
            self.extract_inboard_crossings(file_index, file_name)

    def extract_bboxes(self, file_index, file_name):
        """Extractor which creates whole board description in TF-record format"""

        # Prepare the writer
        self.ds_writers['bboxes'].set_image(file_index,
                                            file_name,
                                            self.board.image,
                                            len(self.board.all_stones))

        # Generate .names and label map files
        fn_map = Path(self.dirs['bboxes']).joinpath('go_board.names')
        if not fn_map.exists():
            with open(str(fn_map), 'w') as f_map:
                f_map.write('black\nwhite\n')
                f_map.close()

        fn_map = Path(self.dirs['bboxes']).joinpath('go_board.pbtxt')
        if not fn_map.exists():
            with open(str(fn_map), 'w') as f_map:
                f_map.write('item {\n\tid: 1\n\tname: \'black\'\n}\n' + \
                            'item {\n\tid: 2\n\tname: \'white\'\n}\n')
                f_map.close()

        # Resize board
        if min(self.n_resize) > 0:
            self.board.resize_board(self.n_resize)

        # Save stones
        for stone in self.board.all_stones:
            area = self.extract_stone_area(stone)
            if area is not None:
                label = 'black' if stone[GR_BW] == STONE_BLACK else 'white'
                self.ds_writers['bboxes'].write_area(area, label)

        # Finalize
        self.ds_writers['bboxes'].finish_image()
        self.add_count('bboxes')

    def one_file(self, file_index, file_name):
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
            extractor_fn(file_index, file_name)

        for k in self.counts:
            if k in self.totals:
                self.totals[k] += self.counts[k]
            else:
                self.totals[k] = self.counts[k]

    def get_args(self):
        parser = ArgumentParser()
        parser.add_argument('pattern', help = 'Selection pattern')
        parser.add_argument('-p', '--positive',
            help = 'Directory to store positives (images with stones) dataset')
        parser.add_argument('-n', '--negative',
            help = 'Directory to store negatives (images without stones) dataset')
        parser.add_argument('-s', '--stones',
            help = "Directory to store stones dataset (stone images, separately for black and white)")
        parser.add_argument('-c', '--crossings',
            help = "Directory to store line crossings and edges dataset (images of board grid lines crossings, " + \
                    "separately for edges, borders crossings and grid lines crossings)")
        parser.add_argument('-b', '--bboxes',
            help = "Directory to store bboxes dataset (one file describing all boards)")
        parser.add_argument('-f', '--format',
            choices=['txt', 'json', 'tf'], default = 'txt',
            help="Output dataset format")
        parser.add_argument('-m', '--method',
            choices = ["single", "enclosed", "both"], default = "both",
            help = "Stone image extration method (for all datasets except bboxes), one of: " + \
                "single - extract areas of single-staying stones, " + \
                "enclosed - extract areas of stones enclosed by other stones, " + \
                "both - extract all stones")
        parser.add_argument('--space',
            nargs = '*',
            default = [10, 3, 5],
            help = "Space to add when extracting area for: single stones, " + \
                    "enclosed stones, edges/crossings " + \
                    "(numbers or perecentage of stone size followed by %)")
        parser.add_argument('--neg-img', type=int,
            default = 0,
            help = 'Number of negative images to generate from one image (0 - the same number as positives)')
        parser.add_argument('--resize', type=int,
            nargs='*',
            default=[0, 0],
            help='Resize images to specified size (0 - no resizing)')
        parser.add_argument('--no-grid',
            action="store_true",
            default = False,
            help = 'Do not generate grid line crossing images')
        parser.add_argument('--rotate',
            type=int,
            nargs=2,
            default=[0, 0],
            help='Two numbers specifying how many rotation images shall be created and an angle for each rotation')
        parser.add_argument('--split',
            type=float,
            help="A float value setting dataset split to train/test datasets")
        parser.add_argument('--shuffle',
            action='store_true',
            help="If True, file list is shuffled before splitting (by default True if split is specified)")

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

        self.split = args.split
        self.shuffle = args.shuffle
        if self.shuffle is None:
            self.shuffle = self.split is not None and self.split > 0

        self.format = args.format

        if self.format == 'tf' and not TF_AVAIL:
            print('TF-record output requested, but Tensorflow is not available. Switching to text output')
            self.format = 'json'
        if self.format == 'txt' and args.bboxes is not None:
            raise ValueError('Cannot generate bboxes dataset in text output format')


    def main(self):
        try:
            self.get_args()
        except:
            print('ERROR:', sys.exc_info()[1])
            return

        # Clean up target directories
        dir_list = [x for x in self.dirs.values() if x is not None]
        def recursive_delete(f):
            if f.is_file(): f.unlink()
            else:
                for x in f.glob('*'):
                    recursive_delete(x)

        for d in dir_list:
            pd = Path(d)
            pd.mkdir(exist_ok=True, parents=True)
            recursive_delete(pd)

        # Make pattern ready for glob:
        # Check it is a directory and if yes, add wildcards
        # If not, check for file wildcards, if none - add them
        if os.path.isdir(self.pattern):
            self.pattern = os.path.join(self.pattern, "*.*")
        else:
            head, tail = os.path.split(self.pattern)
            if tail == '': pattern = os.path.join(self.pattern, "*.*")

        # Load all files
        file_list = []
        print('Counting files...')
        for x in glob.iglob(self.pattern):
            if os.path.isfile(x):
                if Path(x).suffix != '.gpar':
                    # Image files processed as is
                    file_list.append(str(x))
                else:
                    # For .gpar files, try to find an image
                    found = False
                    for sx in ['.png', '.jpg', '.jpeg']:
                        f = Path(x).with_suffix(sx)
                        found = f.exists()
                        if found:
                            file_list.append(str(f))
                            break

                    if not found:
                        print("==> Cannot find an image which corresponds to {} param file".format(x))

        # Shuffle, if requested
        if self.shuffle:
            print('Shuffling file list')
            shuffle(file_list)

        # Prepare stats
        self.file_count = len(file_list)
        self.totals, self.counts = {}, {}

        # Get dataset writers
        if self.format == 'txt':
            writer_class = TxtDatasetWriter
            print('Using text output format')
        elif self.format == 'tf':
            writer_class = TfDatasetWriter
            print('Using TF-record output format')
        else:
            raise ValueError('Dont''t know how to handle dataset format ' + self.format)

        self.ds_writers = {k: writer_class(v, self.file_count, self.split) for k, v in self.dirs.items() if v is not None}

        # Process files
        for n, x in enumerate(file_list):
            self.one_file(n, x)

        # Finalize
        for w in self.ds_writers.values():
            w.close()

        # Show statistics
        print("Dataset items created:")
        for k, v in self.totals.items():
            print("\t{}: {}".format(k, v))

if __name__ == '__main__':
    app = DatasetGenerator()
    app.main()
    cv2.destroyAllWindows()

#"C:\Users\kol\Documents\kol\gbr\img\go_board_*.gpar" -b cc/bboxes --bbox-fmt tf --bbox-split 0.2 --resize 416 416

##    it = tf_record_iterator("C:\\Users\\kol\\Documents\\kol\\gbr\\cc\\stones\\go_board.tfrecord")
##    for n, r in enumerate(it):
##        print('==> ' + str(n))
##        print(r)
##        print('')

