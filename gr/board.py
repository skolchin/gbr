#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Go board class
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
import sys
if sys.version_info[0] < 3:
    from grdef import *
    from gr import process_img, generate_board, find_coord
    from utils import gres_to_jgf, jgf_to_gres, resize2
    import dataset as gds
else:
    from gr.grdef import *
    from gr.gr import process_img, generate_board, find_coord
    from gr.utils import gres_to_jgf, jgf_to_gres, resize2
    import gr.dataset as gds

from pathlib import Path
import cv2
import numpy as np
import json
import logging

class GrBoard(object):
    """ Go board """
    def __init__(self, image_file = None, board_shape = None):
        """ Create new instance either for image file or by generation

        Parameters:
            image_file       Name of image file to load
            board_shape      Generated board shape, if no image file is provided

        """
        self._params = DEF_GR_PARAMS.copy()
        self._res = None
        self._img = None
        self._img_file = None
        self._src_img_file = None
        self._gen_board = False

        if image_file is None or image_file == '':
            # Generate default board
            if board_shape is None: board_shape = DEF_IMG_SIZE
            self.generate(shape = board_shape)
        else:
            # Load board from file
            self.load_image(image_file)

    def load_image(self, filename, f_with_params = True, f_process = True):
        """Loads a new image to board

        Parameters:
            f_with_params     If True, image recognition params are loaed from <filename>.JSON file
            f_process         If True, starts image recongition
        """
        # Load image
        logging.info('Loading {}'.format(filename))
        img = cv2.imread(str(filename))
        if img is None:
           logging.error('Image file not found {}'.format(filename))
           raise Exception('Image file not found {}'.format(filename))

        self._gen_board = False
        self._img_file = filename
        self._src_img_file = filename
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
        """Generates a new board image of given shape.
        if source image was processed, displays recognition results on the image.
        Returns generated OpenCV image.
        """
        self._img = generate_board(shape, res = self._res)
        self._img_file = None
        self._gen_board = True

    def save_image(self, filename = None, max_size = None):
        """Saves image under new name. If max_size provided, resizes image before"""
        if self._img is None:
           raise Exception('Image was not loaded')

        if filename is None: filename = self._img_file
        im = self._img
        if not max_size is None: im = resize(im, max_size)

        logging.info('Saving image to {}'.format(filename))
        try:
            cv2.imwrite(str(filename), im)
        except:
            logging.error(sys.exc_info()[1])
            raise

        self._img_file = filename
        self._gen_board = False

    def load_params(self, filename):
        """Loads recognition parameters from specified file (JSON)"""
        p = json.load(open(str(filename)))
        r = dict()
        for key in self._params.keys():
            if key in p:
                self._params[key] = p[key]
                r[key] = p[key]
        return r

    def load_board_info(self, filename, f_use_gen_img = True, path_override = None):
        """Loads board information from specified file (JGF)"""
        jgf = json.load(open(str(filename)))
        self._res = jgf_to_gres(jgf)

        if not f_use_gen_img:
            # Load existing image
            fn = jgf['image_file']
            if not path_override is None:
               fn = str(Path(path_override).joinpath(Path(fn).name))
            self.load_image(fn, f_process = False)
        else:
            # Use generated image
            self._img_file = jgf['image_file']
            max_e = jgf['edges']['1']
            shape = (max_e[0] + 14,max_e[1] + 14)
            self.generate(shape)

    def save_params(self, filename = None):
        """Saves recognition parameters to specified file (JSON)"""
        if filename is None:
            filename = str(Path(self._img_file).with_suffix('.json'))
        with open(filename, "w") as f:
            json.dump(self._params, f, indent=4, sort_keys=True, ensure_ascii=False)
        return filename

    def save_board_info(self, filename = None):
        """Saves board information to specified file (JGF)"""
        if filename is None:
            filename = str(Path(self._img_file).with_suffix('.jgf'))

        jgf = gres_to_jgf(self._res)
        jgf['image_file'] = self._img_file

        with open(filename, "w") as f:
            json.dump(jgf, f, indent=4, sort_keys=True, ensure_ascii=False)
        return filename

    def load_annotation(self, filename, ds_format = None, f_process = True):
        """Loads annotation from specified file and dataset

        Parameters:
            filename        Name of annotation file
            ds_format       Either a dataset format string or a dataset object
            f_process       True if loaded image has to be processed
        """

        # Load annotation data
        ds = GrDataset.getDataset(ds_format)
        fn, src, _, _ = ds.load_annotation(filename)

        # Load image
        if not src is None and src != '': fn = src
        self.load_image(fn, f_process = f_process)

    def save_annotation(self, filename = None, ds_format = None, anno_only = True, stage = "test"):
        """Saves annotation to specified file and dataset

        Parameters:
            filename        Name of annotation file
            ds_format       Either a dataset format string or a dataset object
            anno_only       True to save only annotation file, False to store image to dataset
            stage           If anno_only is False, name of stage where to save the image

        Returns
            file            Name of file annotation was saved to
        """

        # Check parameters
        if self._img is None:
            return None

        # Prepare data
        extra_param = dict()
        extra_param['image_file'] = self._img_file
        extra_param['source_file'] = self._src_img_file

        # Get a dataset and save
        ds = gds.GrDataset.getDataset(ds_format)
        file, _ = ds.save_annotation(board = self, extra_param = extra_param, \
                                        file_name = filename, anno_only = anno_only, \
                                        stage = stage)

        return file

    def process(self):
        """Does recognition of image file loaded to the board"""
        if self._img is None or self._gen_board:
            self._res = None
        else:
            self._res = process_img(self._img, self._params)
            if not self._res is None:
               if self._res[GR_STONES_B] is None: self._res[GR_STONES_B] = np.array([])
               if self._res[GR_STONES_W] is None: self._res[GR_STONES_W] = np.array([])

    def show_board(self, f_black = True, f_white = True, f_det = False, show_state = None):
        """Generates board image. If a source image was processed, plots recognition results onto the image

        Parameters:
            f_black     If True, black stones are displayed. Not used if show_state is provided
            f_white     If True, white stones are displayed. Not used if show_state is provided
            f_det       If True, stone circles are displayed. Not used if show_state is provided
            show_state  A dictionary of display parameters. If provided, overrides all f_xxx parameters

        Returns:
            img         OpenCV image generated
        """
        if not show_state is None:
           f_black = show_state['black']
           f_white = show_state['white']
           f_det = show_state['box']

        r = None
        if not self._res is None:
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

    def resize_board(self, max_size):
        def resize_stones(stones, scale):
            ret_stones = []
            for st in stones:
                st[GR_X] = int(st[GR_X] * scale[0])
                st[GR_Y] = int(st[GR_Y] * scale[1])
                st[GR_R] = int(st[GR_R] * max(scale[0],scale[1]))
                ret_stones.append(st)
            return np.array(ret_stones)

        self._img, scale = resize2(self._img, max_size)
        if not self._res is None:
            self._res[GR_STONES_B] = resize_stones(self._res[GR_STONES_B], scale)
            self._res[GR_STONES_W] = resize_stones(self._res[GR_STONES_W], scale)
            self._res[GR_SPACING] = (self._res[GR_SPACING][0] * scale[0], \
                                        self._res[GR_SPACING][1] * scale[1])
            self._res[GR_EDGES] = ((self._res[GR_EDGES][0][0] * scale[0], \
                                        self._res[GR_EDGES][0][1] * scale[1]), \
                                        (self._res[GR_EDGES][1][0] * scale[0], \
                                        self._res[GR_EDGES][1][1] * scale[1]))


    @property
    def params(self):
        """Recognition parameters"""
        return self._params

    @params.setter
    def params(self, p):
        """Recognition parameters"""
        for key in p.keys():
            if key in self._params:
                self._params[key] = p[key]

    @property
    def area_mask(self):
        """Board recognition area mask"""
        if 'AREA_MASK' in self._params and type(self._params['AREA_MASK']) is list:
           return self._params['AREA_MASK']
        else:
           return [0, 0, self._img.shape[CV_WIDTH], self._img.shape[CV_HEIGTH]]

    @area_mask.setter
    def area_mask(self, mask):
        self._params['AREA_MASK'] = mask

    @property
    def results(self):
        """Recognition results"""
        return self._res

    @property
    def image(self):
        """Board image"""
        return self._img

    @property
    def image_file(self):
        """Image file name"""
        return self._img_file

    @property
    def is_gen_board(self):
        """True if board was generated with generate()"""
        return self._gen_board

    @property
    def black_stones(self):
        """List of black stones"""
        if self._res is None:
            return None
        else:
            return self._res[GR_STONES_B]

    @property
    def white_stones(self):
        """List of white stones"""
        if self._res is None:
            return None
        else:
            return self._res[GR_STONES_W]

    @property
    def stones(self):
        """Dictionary with all stones (keys are B, W)"""
        if self._res is None:
            return None
        else:
            return { 'W': self._res[GR_STONES_W], 'B': self._res[GR_STONES_B] }

    @property
    def debug_images(self):
        """Collection of debug images generated during image recognition"""
        if self._res is None:
            return None
        else:
            r = dict()
            for key in self._res.keys():
                if key.find("IMG_") >= 0: r[key] = self._res[key]
            return r

    @property
    def debug_info(self):
        """Collection of textual information generated during image recognition"""
        if self._res is None:
            return None
        else:
            r = dict()
            r[GR_EDGES] = self._res[GR_EDGES]
            r[GR_SPACING] = (round(self._res[GR_SPACING][0],2), \
                             round(self._res[GR_SPACING][1],2))
            r[GR_NUM_CROSS_H] = self._res[GR_NUM_CROSS_H]
            r[GR_NUM_CROSS_W] = self._res[GR_NUM_CROSS_W]
            r[GR_BOARD_SIZE] = self._res[GR_BOARD_SIZE]
            r[GR_IMAGE_SIZE] = self._res[GR_IMAGE_SIZE]
            return r

    @property
    def board_shape(self):
        """Board image shape"""
        if self._img is None:
            return None
        else:
            return self._img.shape

    @property
    def board_size(self):
        """Board size"""
        if self._res is None:
            return None
        else:
            return self._res[GR_BOARD_SIZE]

