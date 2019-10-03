#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network MS COCO dataset
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
    from utils import gres_to_jgf
else:
    from gr.board import GrBoard
    from gr.dataset import *
    from gr.utils import gres_to_jgf

class GrCocoDataset(GrDataset):
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        GrDataset.__init__(self, src_path, ds_path, img_size)

        self.meta_path = ensure_path(self.ds_path,"data", "annotations")
        self.img_path = ensure_path(self.ds_path,"data", "coco")

        self.load_metadata()

    @property
    def name(self):
        """Dataset directory name"""
        return "gbr_coco"

    @property
    def ds_format(self):
        """Dataset format"""
        return DS_FMT_COCO

    @property
    def anno_ext(self):
        """Meta file extension"""
        return '.json'


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

            category_id = DS_CLASSES[cls]
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

                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": n,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)

            return bbox

        # Get file name
        meta_file = file_name
        if meta_file is None:
            meta_file = Path(board.image_file).with_suffix('.json')
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
        image_id = get_filename_as_int(filename)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"] = image

        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'

        # Save stones


    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    #os.makedirs(os.path.dirname(json_file), exist_ok=True)
    #json_fp = open(json_file, "w")
    #json_str = json.dumps(json_dict)
    #json_fp.write(json_str)
    #json_fp.close()
    with open(json_file, "w") as f:
        json.dump(json_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
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
        pass

    def save_metadata(self):
        """Saves dataset metadata"""
        pass

    def get_stage(self, file_name):
        """Returns stage where given file belongs to"""
        pass

