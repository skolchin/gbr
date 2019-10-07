#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     GBR dataset base (abstract) class
#
# Author:      kol
#
# Created:     06-09-2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

from pathlib import Path

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
        raise NotImplementedError("Calling of abstract method")

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
        raise NotImplementedError("Calling of abstract method")


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


