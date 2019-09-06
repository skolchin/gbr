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

import sys
if sys.version_info[0] < 3:
    from board import GrBoard
    from dataset import *
else:
    from gr.board import GrBoard
    from gr.dataset import *

class GrPascalDataset(GrDataset):
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        GrDataset.__init__(self, src_path, ds_path, img_size)

        self.meta_path = ensure_path(self.ds_path,"data", "Annotations")
        self.img_path = ensure_path(self.ds_path,"data", "Images")
        self.sets_path = ensure_path(self.ds_path,"data", "ImageSets")

    def save_dataset(self):
        # Prepare file list
        file_list = dict()
        file_list['test'] = []
        file_list['train'] = []

        # Process all JGF (board descr) files in source path
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

    def default_name(self):
        return "gbr_ds"

    def save_annotation(self, board, file_name = None, anno_only = True):

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
                r = max_r
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

        # Check params
        if not anno_only and not filename is None:
            raise Exception('Invalid parameters combination')

        # Get file name
        meta_file = filename
        if meta_file is None:
            meta_file = Path(board.image_file).with_suffix('.xml')
            meta_file = Path(self.meta_path).joinpath(meta_file.name)
            meta_file = str(meta_file)

        # Resize and save to dataset directory if needed
        if not anno_only:
            png_file = Path(board.image_file).with_suffix('.png')
            png_file = Path(self.img_path).joinpath(png_file.name)
            #if not self.img_size[stage] is None and self.img_size[stage] > 0:
            #    board.resize_board(self.img_size[stage])
            board.save_image(str(png_file))

        # Image parameters
        (height, width, depth) = board.image.shape

        # Source
        source = ''
        if not jgf is None and 'source_file' in jgf:
            source = jgf['source_file']

        # Write header
        f = open(str(meta_file),'w')
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t<folder>' + "folder" + '</folder>' + '\n'
        f.write(line)
        line = '\t<filename>' + Path(image_file).name + '</filename>' + '\n'
        f.write(line)
        line = '\t<path>' + image_file + '</path>' + '\n'
        f.write(line)
        line = '\t<source>' + source + '</source>' + '\n'
        f.write(line)
        line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
        line += '\t<depth>' + str(depth) + '</depth>\n\t</size>'
        f.write(line)
        line = '\n\t<segmented>Unspecified</segmented>'
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

        return (bb_b, bb_w)



