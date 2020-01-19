#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     18.01.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cc.item_classifier import BoardItemClassifier

BOARD_SIZE = 19
EMPTY_AREA = [10, 10]
MAX_IMG = 60

AUTOTUNE = tf.data.experimental.AUTOTUNE

class BoardBoxesExtractorLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BoardBoxesExtractorLayer, self).__init__(self, kwargs)

        self.bbox = tf.Variable(initial_value=[0,0], name='bbox', trainable=False)
        self.step = tf.Variable(initial_value=[0,0], name='step', trainable=False)
        self.img_list = None


    def build(self, input_shape):
        print("build() with input shape = ", input_shape)

        n_bbox = np.zeros(2, dtype = np.uint8)
        n_step = np.zeros(2, dtype = np.uint8)

        n_bbox[1] = (input_shape[1] - EMPTY_AREA[1]) // BOARD_SIZE
        n_bbox[0] = (input_shape[0] - EMPTY_AREA[0]) // BOARD_SIZE
        n_step[1] = n_bbox[1] - 5
        n_step[0] = n_bbox[0] - 5

        self.bbox.assign(n_bbox)
        self.step.assign(n_step)

        print("Bbox: {}, step: {}".format(n_bbox, n_step))

    def __call__(self, inputs, *args, **kwargs):
        def get_area(src, x, y, width, height):
            wx = x + width
            wy = y + height
            res = np.empty((height, width, src.shape[2]), dtype=src.dtype)
            res[:] = src[y:wy, x:wx]
            return res

        print("I've been called with ", inputs.shape, inputs.dtype)

        n_bbox = self.bbox.read_value().numpy()
        n_step = self.step.read_value().numpy()

        if any([n_bbox[0] == 0, n_bbox[1] == 0, n_step[0] == 0, n_step[1] == 0]):
            raise ValueError("Invalid parameter")

        np_input = inputs.numpy()
        self.img_list = []
        for y in range(0, max(np_input.shape[0] - n_bbox[0] + 1, MAX_IMG + 1), n_step[0]):
            for x in range(0, np_input.shape[1] - n_bbox[1] + 1, n_step[1]):
                img = get_area(np_input, x, y, n_bbox[1], n_bbox[0])
                self.img_list.extend([img])
        print("Images generated: ", len(self.img_list))

        self.img_list = tf.convert_to_tensor(self.img_list)
        return self.img_list

    def compute_output_shape(self, input_shape):
        if self.img_list is not None:
            return self.img_list.shape
        else:
            n_bbox = self.bbox.read_value().numpy()
            n_step = self.step.read_value().numpy()

            dx = input_shape[1] // n_step[1]
            dy = max(input_shape[0], MAX_IMG) // n_step[0]

            return (dx * dy, n_bbox[0], n_bbox[1], input_shape[2])


class SampleBoardItemClassifier(BoardItemClassifier):

    def __init__(self, model_dir, img_dir, log_dir = None):
        BoardItemClassifier.__init__(self,
            model_dir=model_dir, img_dir=img_dir, img_size=None, log_dir=log_dir)
        self.train_dataset = None
        self.class_names = []

    def load(self):
        """Load a model from directory"""
        print("==> Loading model")
        sample_model = tf.keras.models.load_model(self.model_dir)
        self.model = tf.keras.models.Sequential([BoardBoxesExtractorLayer()])
        for layer in sample_model.layers:
            self.model.add(layer)

    def get_model_layers(self):
        layers = super(SampleBoardItemClassifier, self).get_model_layers()
        layers.insert(0, BoardBoxesExtractorLayer())
        return layers

    def map_fn(self, path):
        """Upload an image fo given path with specified label - internal"""
        image = tf.image.decode_png(tf.io.read_file(path))
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def get_sample_files(self, num_samples = 1):
        img_dir = Path(self.img_dir)
        file_names = [str(f.absolute()) for f in img_dir.glob('*.png')]
        file_names.extend([str(f.absolute()) for f in img_dir.glob('*.jpg')])
        if len(file_names) < num_samples:
            raise Exception('Insufficient number of images in {} ({} required, {} found)'.format(
                            self.img_dir, num_samples, len(file_names)))
        file_names = file_names[0:num_samples]

        return tf.convert_to_tensor(file_names, dtype=tf.string)

    def init_datasets(self, num_samples = 1, display_samples = False):
        file_names = self.get_sample_files(num_samples)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(file_names)
        self.train_dataset = self.train_dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)

        if display_samples:
            self.display_sample_images()

    def predict(self, num_samples = 1, display_predictions = True):
        """Predict on specified number of samples"""
        if self.model is None:
            raise Exception("Model is empty, either build or load it")

        print("==> Prediction")
        file_names = self.get_sample_files(num_samples)
        self.predict_dataset = tf.data.Dataset.from_tensor_slices((file_names))
        self.predict_dataset = self.predict_dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)

        self.predictions = self.model.predict(self.predict_dataset)

        if display_predictions:
            self.display_predictions()



##def display_images(images):
##    plt.figure(figsize=(5,5))
##    for n in range(min(25, images.shape[0])):
##        ax = plt.subplot(5,5,n+1)
##        plt.imshow(images[n])
##        plt.axis('off')
##
##    plt.tight_layout()
##    plt.show()
##
##
###x = tf.image.decode_image(tf.io.read_file('go_board_1.png'))
##
##
##def get_sample_files(img_dir):
##    """Retrieve specified number of sample files from stone images dataset"""
##    img_dir = Path(img_dir)
##    file_names = [str(f.absolute()) for f in img_dir.glob('*.png')]
##    file_names.extend([str(f.absolute()) for f in img_dir.glob('*.jpg')])
##    file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
##
##    return file_names
##
##def map_fn(path):
##    image = tf.image.decode_image(tf.io.read_file(path))
##    image = tf.image.convert_image_dtype(image, tf.float32)
##    return image
##
##file_names = get_sample_files('./')
##dataset = tf.data.Dataset.from_tensor_slices(file_names)
##dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
##
####for _, x in dataset.enumerate():
####    bb = BoardBoxesExtractorLayer()
####    bb.build(x.shape)
####
####    print(bb.compute_output_shape(x.shape))
####
####    y = bb(x)
####    print(y.shape)
####    display_images(y)
##
