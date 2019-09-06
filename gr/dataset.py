#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     GBR dataset base class
#
# Author:      kol
#
# Created:     06-09-2019
# Copyright:   (c) kol 2019
#-------------------------------------------------------------------------------

from pathlib import Path

DS_STAGE = [ 'test', 'train' ]                      # Dataset split
DS_DEF_IMG_SIZE =  { 'test': 0, 'train': 2048 }     # Size of images in dataset
DS_FMT_PASCAL = "pascal"                            # PASCAL VOC dataset format


def ensure_path(base_path, *args):
    """Checks whethe given path exists and creates if not"""
    p = Path(base_path).joinpath(*args)
    if not p.exists(): p.mkdir(parents = True)
    return str(p)


class GrDataset(object):
    """Base class for GBR datasets. load_ and save_ methods shall be considered abstract"""
    def __init__(self, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        """Base class constructor.

        Parameters:
            src_path    path to source images
            ds_path     root path for dataset structure
            img_size    size of images to store in dataset
        """
        root_path = Path(__file__).parent.parent.resolve()
        if ds_path is None:
            if not src_path is None:
                ds_path = str(Path(src_path).parent.resolve().joinpath(self.default_name()))
            else:
                ds_path = str(root_path.joinpath(self.default_name()))

        if src_path is None:
            src_path = str(root_path.joinpath("img"))

        self.src_path = src_path
        self.ds_path = ensure_path(ds_path)
        self.img_size = img_size

    def save_annotation(self, board, file_name = None, anno_only = True):
        """Saves annotation of GrBoard object

        Parameters:
            board       GrBoard with or without recognition results
            anno_only   If True, only annotation file created, otherwise all dataset-related work is performed
            file_name   Name of file to save annotation to.
                        If omitted, when is constructed from board.image_file according
                        to this dataset schema.

        """
        raise NotImplementedError("Calling of abstract method")

    def save_dataset(self):
        """Process src_path and makes annotation for all images there"""
        raise NotImplementedError("Calling of abstract method")

    def default_name(self):
        raise NotImplementedError("Calling of abstract method")

    # Static methods
    __fmt = None

    @staticmethod
    def addFormat(fmt, type_):
        if GrDataset.__fmt is None: GrDataset.__fmt = dict()
        GrDataset.__fmt[fmt] = type_

    @staticmethod
    def getDataset(format, src_path = None, ds_path = None, img_size = DS_DEF_IMG_SIZE):
        """Returns a dataset of given format (see DS_FMT_xxx)"""
        if format is None: format = DS_FMT_PASCAL
        if format in GrDataset.__fmt:
            return GrDataset.__fmt[format](src_path, ds_path, img_size)
        else:
            raise NotImplementedError('Dataset format {} is not supported'.format(format))
