#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network PASCAL VOC dataset
#
# Author:      kol
#
# Created:     03.08.2019.
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
import cv2
import json
import numpy as np
import os
import logging
from pathlib import Path
import xml.dom.minidom as minidom

import sys
if sys.version_info[0] < 3:
    from board import GrBoard
    from dataset import *
    from grdef import *
else:
    from gr.board import GrBoard
    from gr.dataset import *
    from gr.grdef import *

class GrPascalDataset(GrDataset):
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        GrDataset.__init__(self, src_path, ds_path, img_size)

        self.meta_path = ensure_path(self.ds_path,"data", "Annotations")
        self.img_path = ensure_path(self.ds_path,"data", "Images")
        self.sets_path = ensure_path(self.ds_path,"data", "ImageSets")

        self.load_metadata()

    @property
    def name(self):
        """Dataset directory name"""
        return "gbr_ds"

    @property
    def ds_format(self):
        """Dataset format name"""
        return DS_FMT_PASCAL

    @property
    def anno_ext(self):
        """Meta file extension"""
        return '.xml'

    def annotate_board(self, file, jgf, meta_file, image_file, source_file, image_shape):
        """Internal method to generate annotation for board"""
        line = "<annotation>" + '\n'
        file.write(line)
        line = '\t<folder>' + "folder" + '</folder>' + '\n'
        file.write(line)
        line = '\t<filename>' + Path(image_file).name + '</filename>' + '\n'
        file.write(line)
        line = '\t<path>' + image_file + '</path>' + '\n'
        file.write(line)
        line = '\t<source>' + source_file + '</source>' + '\n'
        file.write(line)
        line = '\t<size>\n'
        line += '\t\t<width>'+ str(image_shape[CV_WIDTH]) + '</width>\n'
        line += '\t\t<height>' + str(image_shape[CV_HEIGTH]) + '</height>\n'
        line += '\t\t<depth>' + str(image_shape[CV_CHANNEL]) + '</depth>\n'
        line += '\t</size>\n'
        file.write(line)
        line = '\t<segmented>Unspecified</segmented>'
        file.write(line)
        return None

    def annotate_stones(self, file, jgf, meta_file, image_shape, stone_class):
        """Internal method to generate annotation for stones"""
        if not stone_class in jgf: return None

        stones = jgf[stone_class]
        if stones is None or len(stones) == 0: return None
        bbox = np.empty((len(stones),5), dtype = np.float)

        # Proceed
        n = 0
        for i in stones:
            x = stones[i]['X']
            y = stones[i]['Y']
            r = stones[i]['R']
            pos = stones[i]['A'] + stones[i]['B']

            a = r + 2
            xmin = max(x - int(a), 1)
            ymin = max(y - int(a), 1)
            xmax = min(x + int(a), image_shape[CV_WIDTH])
            ymax = min(y + int(a), CV_HEIGTH)

            bbox[n,0] = xmin
            bbox[n,1] = ymin
            bbox[n,2] = xmax
            bbox[n,3] = ymax
            bbox[n,4] = 100.0
            n += 1

            line = '\n\t<object>'
            line += '\n\t\t<name>' + stone_class + '</name>'
            line += '\n\t\t<pose>Unspecified</pose>'
            line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>Unspecified</difficult>'
            line += '\n\t\t<position>' + pos + '</position>'

            line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
            line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
            line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
            line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
            line += '\n\t\t</bndbox>'

            line += '\n\t</object>'

            file.write(line)

        return bbox

    def annotation_close(self, file, jgf, meta_file):
        """Internal method to close annotation"""
        line = "\n</annotation>" + '\n'
        file.write(line)

    def load_annotation(self, file_name):
        """Loads annotation from given file

        Parameters:
            file_name   Name of file to load annotation from

        Returns:
            image_file  Name of image file specified in annotaion
            src_file    Name of annotation's source file (if set)
            shape       Tuple of image shape (height, width, depth)
            bboxes      List of tuples specifying bounding boxes:
                             (x1,y1) - top-left corner
                             (x2,y2) - bottom-right corner
                             class   - object class (black/white)
        """

        def get_tag(node, tag):
            d = node.getElementsByTagName(tag)
            if d is None: return None
            else:
                d = d[0].firstChild
                if d is None: return None
                else: return d.data

        def get_child_node(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        # Load annotation file
        with open(file_name) as f:
            data = minidom.parseString(f.read())

        # Find image file name
        image_file = get_tag(data, 'path')
        logging.info('Image file: {}'.format(image_file))
        if not Path(image_file).is_file():
            image_file = str(self.src_path.joinpath(Path(image_file).name))
            logging.info('Image file not found, overriding with default image path')

        # Find source file name
        src_file = get_tag(data, 'source')
        if src_file is None:
           logging.info("Source file name not defined")
        else:
           logging.info('Source file: {}'.format(src_file ))

        # Find image shape
        size = data.getElementsByTagName('size')
        if size is None or len(size) == 0:
           shape = None
           logging.info("Image size not defined")
        else:
            height = int(get_child_node(size[0], 'height'))
            width = int(get_child_node(size[0], 'width'))
            depth = int(get_child_node(size[0], 'depth'))
            shape = [width, height, depth]
            logging.info('Image shape: {}, {}, {}'.format(width, height, depth))

        # Load and parse the objects
        objs = data.getElementsByTagName('object')
        logging.info('Objects count: {}'.format(len(objs)))
        bboxes = []
        for ix, obj in enumerate(objs):
            # Get coordinates and class
            x1 = int(get_child_node(obj, 'xmin'))
            y1 = int(get_child_node(obj, 'ymin'))
            x2 = int(get_child_node(obj, 'xmax'))
            y2 = int(get_child_node(obj, 'ymax'))
            cls = str(get_child_node(obj, "name")).lower().strip()
            logging.info('Class {} object ({},{}) - ({},{})'.format(cls, x1,y1, x2,y2))

            # Check coordinates
            if x1 <= 0 or y1 <= 0 or x1 >= shape[1] or y1 >= shape[0]:
                logging.error("Point {} coordinates out of boundaries ({},{})-({},{}) <> ({},{})".format(ix, x1, \
                    y1, x2, y2, shape[1], shape[0]))
            if x1 >= x2 or y1 >= y2:
                logging.error("Coordinates ({},{}) and ({},{}) overlap".format(x1, y1, x2, y2))

            # Save bounding box
            bboxes.append(((x1,y1), (x2,y2), cls))

        return image_file, src_file, shape, bboxes

    def load_metadata(self):
        """Loads dataset metadata"""
        for ds in DS_STAGE:
            fn = Path(self.sets_path).joinpath(ds + '.txt')
            try:
                with open(str(fn), 'r') as f:
                    self._stage_files[ds] = f.read().splitlines()
                    f.close()
            except:
                self._stage_files[ds] = []

    def save_metadata(self):
        """Saves dataset metadata"""
        for ds in ('test', 'train'):
            fn = Path(self.sets_path).joinpath(ds + '.txt')
            try:
                with open(str(fn), 'w') as f:
                    f.writelines( "%s\n" % item for item in self._stage_files[ds])
                    f.close()
            except:
                self._stage_files[ds] = []

    def get_stage(self, file_name):
        """Returns stage where given file belongs to"""
        fn = Path(file_name).stem
        return GrDataset.get_stage(self, str(fn))

