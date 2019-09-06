#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network PASCAL VOC dataset
#
# Author:      skolchin
#
# Created:     03.08.2019.
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
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
    from utils import gres_to_jgf
else:
    from gr.board import GrBoard
    from gr.dataset import *
    from gr.utils import gres_to_jgf

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

    def ds_format(self):
        """Dataset format name"""
        return DS_FMT_PASCAL

    @property
    def anno_ext(self):
        """Meta file extension"""
        return '.xml'

    def generate_dataset(self):
        """Regenerates a dataset"""

        # Prepare file list
        file_list = dict()
        file_list['test'] = []
        file_list['train'] = []

        # Process all JGF (board descr) files in source path
        try:
            for file in os.listdir(self.src_path):
                src_file = os.path.join(self.src_path, file)
                board = GrBoard()
                stage = ""
                max_size = None

                if file.endswith('.jgf'):
                    # Process JGF file. Images with JGF would go training dataset
                    board.load_board_info(src_file, f_use_gen_img = False, path_override = self.src_path)
                    stage = "train"
                elif file.endswith('.jpg') or file.endswith('.png'):
                    # Find JGF. Images without board info will be used in testing dataset
                    jgf_file = os.path.splitext(src_file)[0] + '.jgf'
                    if os.path.exists(jgf_file): continue

                    # Process image file
                    logging.info ("Loading file {} to testing dataset".format(src_file))
                    board.load_image(src_file, f_process = False)
                    stage = "test"
                else:
                    continue

                logging.info ("Appending to dataset {}".format(stage))

                # Convert to PNG
                image_file = os.path.basename(board.image_file)
                png_file = os.path.splitext(os.path.join(self.img_path,image_file))[0] + '.png'
                if not self.img_size[stage] is None and self.img_size[stage] > 0:
                    board.resize_board(self.img_size[stage])
                board.save_image(str(png_file))

                # Save annotation
                meta_file = os.path.splitext(os.path.join(self.meta_path,image_file))[0] + '.xml'
                board.save_annotation(meta_file)

                # Add to file list
                file_list[stage].append(os.path.splitext(os.path.basename(png_file))[0])

            # Save datasets
            for mode in file_list:
                ds_name = os.path.join(self.sets_path, mode+'.txt')
                logging.info("Creating dataset {}".format(ds_name))
                count = 0
                with open(str(ds_name), "w+") as f:
                    for file in file_list[mode]:
                        f.write("{}\n".format(file))
                        count += 1
                    f.close()
                logging.info("{} entries written".format(count))
        except:
            logging.exception('Error in GrPascalDataset.save_dataset()')
            raise


    def save_annotation(self, board, extra_param = None, file_name = None, anno_only = True, stage = "train"):
        """Saves annotation of GrBoard instance

        Parameters:
            board       GrBoard with or without recognition results
            file_name   Name of file to save annotation to.
                        If omitted, when is constructed from board.image_file according to this dataset schema
            anno_only   If True, only annotation file created, otherwise all dataset-related work is performed
                        Cannot be True with file_name is not None
            stage       Stage to save annotation for

        Returns:
            filename    Name of file annotation was saved to
            bbox        Tuple of bounding boxes for black and white stones

        Note that if board is saved to dataset (anno_only == false), board will be resized
        to image size specified upon dataset construction for given stage
        """

        def annotate_stones(file, jgf, shape, cls):
            # Init
            if not cls in jgf:
                return

            stones = jgf[cls]
            if stones is None or len(stones) == 0:
               return
            bbox = np.empty((len(stones),5), dtype = np.float)

            # Find radius which appears most often
            rlist = [f[1]['R'] for f in stones.items()]
            unique, counts = np.unique(rlist, return_counts=True)
            summary = dict(zip(unique, counts))
            max_r = max(summary, key = lambda x: summary[x])

            # Proceed
            n = 0
            for i in stones:
                x = stones[i]['X']
                y = stones[i]['Y']
                r = stones[i]['R']
                #r = max_r
                pos = stones[i]['A'] + stones[i]['B']

                a = r + 2
                xmin = x - int(a)
                if xmin <= 0: xmin = 1
                ymin = y - int(a)
                if ymin <= 0: ymin = 1
                xmax = x + int(a)
                if xmax > shape[1]: xmax = shape[1]
                ymax = y + int(a)
                if ymax > shape[0]: ymax = shape[0]

                bbox[n,0] = xmin
                bbox[n,1] = ymin
                bbox[n,2] = xmax
                bbox[n,3] = ymax
                bbox[n,4] = 100.0
                n += 1

                line = '\n\t<object>'
                line += '\n\t\t<name>' + cls + '</name>'
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

        # Get file name
        meta_file = file_name
        if meta_file is None:
            meta_file = Path(board.image_file).with_suffix('.xml')
            meta_file = Path(self.meta_path).joinpath(meta_file.name)
            meta_file = str(meta_file)
        logging.info('Saving annotation to {}'.format(meta_file))

        # Resize and save image to dataset directory if requested
        if not anno_only:
           st = self.get_stage(board.image_file)
           logging.info('Stages: {} -> {}'.format(st, stage))
           png_file = Path(board.image_file).with_suffix('.png')
           png_file = Path(self.img_path).joinpath(png_file.name)
           logging.info('Saving image to {}'.format(png_file))

           if not self.img_size[stage] is None and self.img_size[stage] > 0:
              logging.info('Resizing to {}'.format(self.img_size[stage]))
              board.resize_board(self.img_size[stage])

           board.save_image(str(png_file))

        # Image parameters
        (height, width, depth) = board.image.shape

        # Setup jgf dictionary
        jgf = None
        if not board.results is None:
            jgf = gres_to_jgf(board.results)
        else:
            jgf = dict()
        if not extra_param is None:
           for k in extra_param.keys():
               jgf[k] = extra_param[k]

        # Find image source
        source = ''
        if not jgf is None and 'source_file' in jgf:
            source = jgf['source_file']

        # Write header
        f = open(str(meta_file),'w')
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t<folder>' + "folder" + '</folder>' + '\n'
        f.write(line)
        line = '\t<filename>' + Path(board.image_file).name + '</filename>' + '\n'
        f.write(line)
        line = '\t<path>' + board.image_file + '</path>' + '\n'
        f.write(line)
        line = '\t<source>' + source + '</source>' + '\n'
        f.write(line)
        line = '\t<size>\n'
        line += '\t\t<width>'+ str(width) + '</width>\n'
        line += '\t\t<height>' + str(height) + '</height>\n'
        line += '\t\t<depth>' + str(depth) + '</depth>\n'
        line += '\t</size>\n'
        f.write(line)
        line = '\t<segmented>Unspecified</segmented>'
        f.write(line)

        # Write objects (stones)
        bb_b = None
        bb_w = None
        if not jgf is None:
            bb_b = annotate_stones(f, jgf, (height, width), 'black')
            bb_w = annotate_stones(f, jgf, (height, width), 'white')

        # Close
        line = "\n</annotation>" + '\n'
        f.write(line)
        f.close()

        # Update dataset metadata
        if not anno_only:
           cur_st = self.get_stage(board.image_file)
           if cur_st is None:
              self._stage_files[stage].append(str(png_file.stem))
              self.save_metadata()
              logging.info('Image added to {} stage'.format(stage))
           elif cur_st != stage:
               self._stage_files[cur_st].remove(str(png_file.stem))
               self._stage_files[stage].append(str(png_file.stem))
               self.save_metadata()
               logging.info('Image moved to {} stage'.format(stage))

        return meta_file, (bb_b, bb_w)


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
                self.datasets[ds] = []

    def save_metadata(self):
        """Saves dataset metadata"""
        for ds in ('test', 'train'):
            fn = Path(self.sets_path).joinpath(ds + '.txt')
            try:
                with open(str(fn), 'w') as f:
                    f.writelines( "%s\n" % item for item in self._stage_files[ds])
                    f.close()
            except:
                self.datasets[ds] = []

    def get_stage(self, file_name):
        """Returns stage where given file belongs to"""
        fn = Path(file_name).stem
        return GrDataset.get_stage(self, str(fn))

