#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go board class
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------
import sys
if sys.version_info[0] < 3:
    from grdef import *
    from gr import process_img, generate_board, find_coord
    from utils import gres_to_jgf, jgf_to_gres, resize
    from net_utils import make_anno
else:
    from gr.grdef import *
    from gr.gr import process_img, generate_board, find_coord
    from gr.utils import gres_to_jgf, jgf_to_gres, resize
    from gr.net_utils import make_anno

import xml.dom.minidom as minidom

from pathlib import Path
import cv2
import numpy as np
import json

class GrBoard(object):
    def __init__(self, image_file = None, board_shape = DEF_IMG_SIZE):
        self._params = DEF_GR_PARAMS.copy()
        self._res = None
        self._img = None
        self._img_file = None
        self._gen_board = False

        if image_file is None or image_file == '':
            # Generate default board
            self.generate(shape = board_shape)
        else:
            # Load board from file
            self.load_image(image_file)

    def load_image(self, filename, f_with_params = True, f_process = True):
        # Load image
        img = cv2.imread(str(filename))
        if img is None:
            raise Exception('Image file not found {}'.format(filename))
        self._gen_board = False
        self._img_file = filename
        self._img = img

        # Load params, if requested and file exists
        f_params_loaded = False
        if f_with_params:
            params_file = Path(filename).with_suffix('.json')
            if params_file.is_file():
                self.load_params(str(params_file))
                f_params_loaded = True

        # Analyze board
        if f_process: self.process()
        return f_params_loaded

    def generate(self, shape = DEF_IMG_SIZE):
        self._img = generate_board(shape, res = self._res)
        self._img_file = None
        self._gen_board = True

    def save_image(self, filename = None, max_size = None):
        if self._img is None:
           raise Exception('Image was not loaded')

        if filename is None: filename = self._img_file
        im = self._img
        if not max_size is None: im = resize(im, max_size)

        cv2.imwrite(str(filename), im)

        self._img_file = filename
        self._gen_board = False

    def load_params(self, filename):
        p = json.load(open(str(filename)))
        r = dict()
        for key in self._params.keys():
            if key in p:
                self._params[key] = p[key]
                r[key] = p[key]
        return r

    def load_board_info(self, filename, f_use_gen_img = True, image_path = None):
        jgf = json.load(open(str(filename)))
        self._res = jgf_to_gres(jgf)

        if not f_use_gen_img:
            # Load existing image
            fn = jgf['image_file']
            if not Path(fn).is_file() and not image_path is None:
               # W/A to replace wrong path to the image
               fn = str(Path(image_path).joinpath(Path(fn).name))
            self.load_image(fn, f_process = False)
        else:
            # Use generated image
            self._img_file = jgf['image_file']
            max_e = jgf['edges']['1']
            shape = (max_e[0] + 14,max_e[1] + 14)
            self.generate(shape)

    def save_params(self, filename = None):
        if filename is None:
            filename = str(Path(self._img_file).with_suffix('.json'))
        with open(filename, "w") as f:
            json.dump(self._params, f, indent=4, sort_keys=True, ensure_ascii=False)
        return filename

    def save_board_info(self, filename = None):
        if filename is None:
            filename = str(Path(self._img_file).with_suffix('.jgf'))

        jgf = gres_to_jgf(self._res)
        jgf['image_file'] = self._img_file

        with open(filename, "w") as f:
            json.dump(jgf, f, indent=4, sort_keys=True, ensure_ascii=False)
        return filename

    def load_annotation(self, filename):
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        # Load annotation file
        with open(filename) as f:
            data = minidom.parseString(f.read())

        # Find image file name
        fn_list = data.getElementsByTagName('path')
        if fn_list is None or len(fn_list) == 0:
            raise Exception('Filename not specified')

        # Load image
        fn = fn_list[0].firstChild.data
        self.load_image(fn, f_process = True)

    def save_annotation(self, filename = None, max_size = None):
        if self._img is None:
            return None
        if filename is None:
            filename = str(Path(self._img_file).with_suffix('.xml'))

        im = self._img
        if not max_size is None: im = resize(im, max_size)

        jgf = None
        if not self._res is None: jgf = gres_to_jgf(self._res)
        make_anno(filename, self._img_file, img = im, jgf = jgf)

        return filename

    def process(self):
        if self._img is None or self._gen_board:
            self._res = None
        else:
            self._res = process_img(self._img, self._params)

    def show_board(self, f_black = True, f_white = True, f_det = False, f_anno = False):
        r = self._res.copy()
        if not f_black:
            del r[GR_STONES_B]
        if not f_white:
            del r[GR_STONES_W]

        img = generate_board(shape = self._img.shape, res = r, f_show_det = f_det)
        return img

    def find_stone(self, coord = None, pos = None):
        if not coord is None:
            c = None
            pt = find_coord(coord[0], coord[1], self.black_stones)
            if not pt is None:
                c = GR_STONES_B
            else:
                pt = find_coord(coord[0], coord[1], self.white_stones)
                if not pt is None: c = GR_STONES_W
            return pt, c
        else:
            return None, None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        for key in p.keys():
            if key in self._params:
                self._params[key] = p[key]

    @property
    def results(self):
        return self._res

    @property
    def image(self):
        return self._img

    @property
    def image_file(self):
        return self._img_file

    @property
    def is_gen_board(self):
        return self._gen_board

    @property
    def black_stones(self):
        if self._res is None:
            return None
        else:
            return self._res[GR_STONES_B]

    @property
    def white_stones(self):
        if self._res is None:
            return None
        else:
            return self._res[GR_STONES_W]

    @property
    def stones(self):
        if self._res is None:
            return None
        else:
            return { 'W': self._res[GR_STONES_W], 'B': self._res[GR_STONES_B] }

    @property
    def debug_images(self):
        if self._res is None:
            return None
        else:
            r = dict()
            for key in self._res.keys():
                if key.find("IMG_") >= 0: r[key] = self._res[key]
            return r

    @property
    def debug_info(self):
        if self._res is None:
            return None
        else:
            r = dict()
            r[GR_EDGES] = self._res[GR_EDGES]
            r[GR_SPACING] = self._res[GR_SPACING]
            r[GR_NUM_CROSS_H] = self._res[GR_NUM_CROSS_H]
            r[GR_NUM_CROSS_W] = self._res[GR_NUM_CROSS_W]
            r[GR_BOARD_SIZE] = self._res[GR_BOARD_SIZE]
            r[GR_IMAGE_SIZE] = self._res[GR_IMAGE_SIZE]
            return r

    @property
    def board_shape(self):
        if self._img is None:
            return None
        else:
            return self._img.shape

    @property
    def board_size(self):
        if self._res is None:
            return None
        else:
            return self._res[GR_BOARD_SIZE]

