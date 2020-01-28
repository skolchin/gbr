#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     20.01.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import Iterator

# The routines below are copied from tensorflow source code
# available at https://github.com/keras-team/keras-preprocessing/blob/9a836c25177e1be5940e1b2ab19fdb383225c32a/keras_preprocessing/image/utils.py
# (c) by TensorFlow Authors
def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith('.tiff'):
                warnings.warn('Using ".tiff" files with multiple bands '
                              'will cause distortion. Please verify your output.')
            if fname.lower().endswith(white_list_formats):
                yield root, fname

def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.
    # Returns
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames


class BoardAreaGenerator(ImageDataGenerator):

    def __init__(self, **kwargs):
        super(BoardAreaGenerator,self).__init__(self, kwargs)
        self.classes = ['black', 'white', 'edge', 'border']
        self.split = None

    def flow_from_directory(self, directory, **kwargs):
        follow_links = kwargs.pop('follow_links', False)
        subset = kwargs.pop('subset', None)

        self.num_classes = len(self.classes)
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        if subset is not None:
            validation_split = self._validation_split
            if subset == 'validation':
                self.split = (0, validation_split)
            elif subset == 'training':
                self.split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            self.split = None

        results = []
        for dirpath in (os.path.join(directory, subdir) for subdir in self.classes):
            results.append(
                _list_valid_filenames_in_directory(dirpath,
                                                   Iterator.white_list_formats,
                                                   self.split,
                                                   self.class_indices, follow_links))
        return results


g = BoardAreaGenerator()
f = g.flow_from_directory('C:\\Users\\kol\\Documents\\kol\\gbr\\img')
print(len(f))

