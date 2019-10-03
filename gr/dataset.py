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

import sys
if sys.version_info[0] < 3:
    from board import GrBoard
    from utils import gres_to_jgf
else:
    from gr.board import GrBoard
    from gr.utils import gres_to_jgf

DS_STAGE = [ 'test', 'train' ]                      # Dataset split stages
DS_DEF_IMG_SIZE =  { 'test': 0, 'train': 2048 }     # Size of images in dataset
DS_FMT_PASCAL = "pascal"                            # PASCAL VOC dataset format
DS_FMT_COCO = "coco"                                # MS COCO dataset format
DS_CLASSES = { 'black': 0, 'white': 1 }



def ensure_path(base_path, *args):
    """Checks whethe given path exists and creates if not"""
    p = Path(base_path).joinpath(*args)
    if not p.exists(): p.mkdir(parents = True)
    return str(p)


class GrDataset(object):
    """Base class for GBR datasets. load_ and save_ methods shall be considered abstract"""
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        """Constructor

        Parameters:
            src_path    path to source images
            ds_path     root path for dataset structure
            img_size    size of images to store in dataset
        """

        # Construct paths
        root_path = Path(__file__).parent.parent.resolve()
        if ds_path is None:
            if not src_path is None:
                ds_path = str(Path(src_path).parent.resolve().joinpath(self.name))
            else:
                ds_path = str(root_path.joinpath(self.name))

        if src_path is None:
            src_path = str(root_path.joinpath("img"))

        self.src_path = src_path
        self.ds_path = ensure_path(ds_path)
        self.img_size = img_size

        self.use_image_ids = False
        self.separate_stages = False

        # Prepare metadata
        self._stage_files = dict()
        for k in DS_STAGE: self._stage_files[k] = []

    @property
    def name(self):
        """Name of dataset directory or file"""
        raise NotImplementedError("Calling of abstract method")

    @property
    def ds_format(self):
        """Dataset format name"""
        raise NotImplementedError("Calling of abstract method")

    @property
    def anno_ext(self):
        """Returns dataset annotation files extension. Has to be overriden for file datasets"""
        return None

    @property
    def items(self):
        """Dictionary of files or entries in a dataset (keys are stages if dataset separates file by stage or 'files' otherwise)
        If a dataset doesn't use files, this method has to be overriden
        """
        file_list = []
        g = Path(self.src_path).glob('*' + self.anno_ext)
        for x in g:
            if x.is_file(): file_list.append(str(x))
        return { 'files': file_list }

    @property
    def stage_files(self):
        """Allocation of files or entries to stages"""
        return self._stage_files.copy()

    @stage_files.setter
    def stage_files(self, sf):
        """Allocation of files or entries to stages"""
        for stage in DS_STAGE:
            self._stage_files[stage] =  sf[stage].copy()

    @property
    def image_size(self):
        """Stage image sizes"""
        return self.img_size

    @image_size.setter
    def image_size(self, sz):
        """Stage image sizes"""
        for stage in DS_STAGE:
            self.img_size[stage] =  sz[stage]

    def annotate_board(self, file, jgf, meta_file, image_file, source_file, image_shape):
        """Internal method to generate annotation for board"""
        return None

    def annotate_stones(self, file, jgf, meta_file, image_shape, stone_class):
        """Internal method to generate annotation for stones"""
        return None

    def annotation_close(self, file, jgf, meta_file):
        """Internal method to close annotation"""
        return None


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
        raise NotImplementedError("Calling of abstract method")

    def load_metadata(self):
        """Loads dataset metadata. Should populate self._stage_files"""
        raise NotImplementedError("Calling of abstract method")

    def save_metadata(self):
        """Saves dataset metadata"""
        raise NotImplementedError("Calling of abstract method")


    def get_stage(self, file_name):
        """Returns stage where given file belongs to"""
        for stage in self._stage_files:
            for f in self._stage_files[stage]:
                if f.lower() == file_name.lower():
                   return stage
        return None

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


    # Static methods
    __fmt = None

    @staticmethod
    def addFormat(fmt, type_):
        if GrDataset.__fmt is None: GrDataset.__fmt = dict()
        GrDataset.__fmt[fmt] = type_

    @staticmethod
    def getDataset(ds_format = DS_FMT_PASCAL, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        """Returns a dataset of given format

        Parameters:
            ds_format   Dataset format (see DS_FMT_xxx). DS_FMT_PASCAL is default
            src_path    path to source images
            ds_path     root path for dataset structure
            img_size    size of images to store in dataset (DS_DEF_IMG_SIZE by default)
        """
        if issubclass(type(ds_format), GrDataset):
           return ds_format
        else:
           if ds_format is None: ds_format = DS_FMT_PASCAL
           if ds_format in GrDataset.__fmt:
              return GrDataset.__fmt[ds_format](src_path, ds_path, img_size)
           else:
              raise NotImplementedError('Dataset format {} is not supported'.format(ds_format))

    @staticmethod
    def defaultImgSize():
        return DS_DEF_IMG_SIZE


