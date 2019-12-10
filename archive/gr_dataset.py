#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     GBR dataset base class
#
# Author:      kol
#
# Created:     06-09-2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

from pathlib import Path
import logging
import os

from .board import GrBoard
from .utils import gres_to_jgf
from .dataset import *

class GrDatasetBase(object):

    """A datasets class with some common method implementations"""
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        GrDataset.__init__(self, src_path, ds_path, img_size)

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
        self.annotate_board(f, jgf, meta_file, board.image_file, source, board.image.shape)

        # Write objects (stones)
        bb_b = None
        bb_w = None
        if not jgf is None:
            bb_b = self.annotate_stones(f, jgf, meta_file, board.image.shape, 'black')
            bb_w = self.annotate_stones(f, jgf, meta_file, board.image.shape, 'white')

        # Close
        self.annotation_close(f, jgf, meta_file)
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

    def generate_dataset(self):
        """Regenerates a dataset"""

        # Clear metadata
        for k in DS_STAGE: self._stage_files[k] = []

        # Check dirs
        if self.separate_stages:
            for t in DS_STAGE:
                ensure_path(self.img_path, t)
                ensure_path(self.meta_path, t)

        # Process all JGF (board descr) files in source path
        try:
            for file in os.listdir(self.src_path):
                src_file = os.path.join(self.src_path, file)
                logging.info ("Processing {}".format(src_file))

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
                    board.load_image(src_file, f_process = False)
                    stage = "test"
                else:
                    continue

                # Convert to PNG
                image_file = os.path.basename(board.image_file)
                if self.use_image_ids:
                    # Dataset image name is index wihin the stage
                    png_fname = stage + '_' + str(len(file_list[stage])).zfill(4)
                else:
                    # Dataset image name is source file name
                    png_fname = image_file
                if self.separate_stages:
                    # Create separate directories for each stage
                    png_file = os.path.splitext(os.path.join(self.img_path, stage, png_fname))[0] + '.png'
                else:
                    # Use one directory
                    png_file = os.path.splitext(os.path.join(self.img_path, png_fname))[0] + '.png'

                if not self.img_size[stage] is None and self.img_size[stage] > 0:
                    board.resize_board(self.img_size[stage])

                board.save_image(str(png_file))

                # Save annotation
                if self.use_image_ids:
                    # Dataset image name is index wihin the stage
                    meta_fname = str(len(file_list[stage])).zfill(4)
                else:
                    # Dataset image name is source file name
                    meta_fname = image_file
                if self.separate_stages:
                    # Create separate directories for each stage
                    meta_file = os.path.splitext(os.path.join(self.meta_path, stage, meta_fname))[0] + '.xml'
                else:
                    # Use one directory
                    meta_file = os.path.splitext(os.path.join(self.meta_path, meta_fname))[0] + '.xml'
                board.save_annotation(meta_file)

                # Add to file list
                self._stage_files[stage].append(os.path.splitext(os.path.basename(png_file))[0])

            # Save datasets
            logging.info("Saving metadata")
            self.save_metadata()
        except:
            logging.exception('Error')
            raise


