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
from tensorflow.python.framework import tensor_shape

from cc.item_classifier import BoardItemClassifier

BOARD_SIZE = 19
EMPTY_AREA = [10, 10]
MAX_IMG = 60
DISPLAY_COLS = 6
CONFIDENCE_LEVEL = 0.8

AUTOTUNE = tf.data.experimental.AUTOTUNE

# BoardBoxesExtractor layer class
class BoardBoxesExtractorLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BoardBoxesExtractorLayer, self).__init__(self, kwargs)

        self.bbox = np.array(kwargs.pop('bbox', [0,0]))
        self.stride = np.array(kwargs.pop('stride', [0,0]))
        self.img_list = None


    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        print("build() with input shape = ", input_shape)

        self.bbox = np.zeros(2, dtype = np.uint8)
        self.stride = np.zeros(2, dtype = np.uint8)

        self.bbox[1] = (input_shape[1] - EMPTY_AREA[1]) // BOARD_SIZE
        self.bbox[0] = (input_shape[0] - EMPTY_AREA[0]) // BOARD_SIZE
        self.stride[1] = int(self.bbox[1] * (2/3))
        self.stride[0] = int(self.bbox[0] * (2/3))

        print("bbox: {}, stride: {}".format(self.bbox, self.stride))

    def call(self, inputs, *args, **kwargs):
        def get_area(src, x, y, width, height):
            wx = x + width
            wy = y + height
            res = np.empty((height, width, src.shape[2]), dtype=src.dtype)
            res[:] = src[y:wy, x:wx]
            return res

        input_shape = inputs.get_shape()
        print("I've been called with ", input_shape)

        np_input = inputs.numpy() if len(inputs.shape) == 3 else inputs[0].numpy()
        if self.bbox[0] == 0:
            self.build(np_input.shape)

        if any([self.bbox[0] == 0, self.bbox[1] == 0, self.stride[0] == 0, self.stride[1] == 0]):
            raise ValueError("Invalid parameter")

        self.img_list = []
        for y in range(0, max(np_input.shape[0] - self.bbox[0] + 1, MAX_IMG + 1), self.stride[0]):
            for x in range(0, np_input.shape[1] - self.bbox[1] + 1, self.stride[1]):
                img = get_area(np_input, x, y, self.bbox[1], self.bbox[0])
                self.img_list.extend([img])
        print("Images generated: ", len(self.img_list))

        self.img_list = tf.convert_to_tensor(self.img_list)
        return self.img_list

    def compute_output_shape(self, input_shape):
        if self.img_list is not None:
            return self.img_list.shape
        else:
            dx = input_shape[1] // self.stride[1]
            dy = max(input_shape[0], MAX_IMG) // self.stride[0]
            return (dx * dy, self.bbox[0], self.bbox[1], input_shape[2])


# BoardBoxesExtractor layer class rewritten to use TF ops
class BoardBoxesExtractorLayerV2(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        tf.keras.layers.Layer.__init__(self, trainable=False,
            name='bbox_extractor', **kwargs)

        self.bbox = np.array(kwargs.pop('bbox', [0,0]))
        self.stride = np.array(kwargs.pop('stride', [0,0]))
        self.img_list = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        print("building with input shape = ", input_shape)

        self.bbox = np.zeros(2, dtype = np.uint8)
        self.stride = np.zeros(2, dtype = np.uint8)

        if len(input_shape) == 4: input_shape = input_shape[1:]

        self.bbox[1], self.bbox[0] = 20, 20
        self.stride[1] = self.bbox[1] // 2
        self.stride[0] = self.bbox[0] // 2

        print("bbox: {}, stride: {}".format(self.bbox, self.stride))

    @tf.function
    def call(self, inputs, *args, **kwargs):
        input_shape = inputs.get_shape()
        print("I've been called with ", input_shape)
        if len(inputs.shape) != 4:
            raise ValueError("Input shape must be equal to 4")

        if any([self.bbox[0] == 0, self.bbox[1] == 0, self.stride[0] == 0, self.stride[1] == 0]):
            raise ValueError("Invalid parameter")

        sizes = [1, self.bbox[1], self.bbox[0], 1]
        strides = [1, self.stride[1], self.stride[0], 1]
        self.img_list = tf.image.extract_patches(inputs,
                                                 sizes=sizes,
                                                 strides=strides,
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')

        self.img_list = tf.reshape(self.img_list, (-1, self.bbox[1], self.bbox[0], 3))
        print("Resulting images tensor: ", self.img_list.shape)

        return self.img_list

    def compute_output_shape(self, input_shape):
        if self.img_list is not None:
            return self.img_list.shape
        else:
            input_shape = tensor_shape.TensorShape(input_shape)
            if len(input_shape) == 4: input_shape = input_shape[1:]

            dx = input_shape[1] // self.stride[1]
            dy = max(input_shape[0], MAX_IMG) // self.stride[0]

            return (1, dx * dy, self.bbox[0], self.bbox[1], input_shape[2])


# BoardBoxesExtractor layer class rewritten to use TF ops and variables
class BoardBoxesExtractorLayerV3(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        #kwargs.setdefault('trainable', False)
        #kwargs.setdefault('name', 'bbox_extractor')
        tf.keras.layers.Layer.__init__(self, trainable=True,
            name='bbox_extractor', **kwargs)

        self.n_bbox, self.n_stride = [0,0], [0,0]
        self.bbox = tf.Variable(initial_value=self.n_bbox, name='bbox', trainable=False)
        self.stride = tf.Variable(initial_value=self.n_stride, name='stride', trainable=False)
        self.img_list = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape([None, 487, 487, 3])
        print("building bbox layer with input shape ", input_shape)

        if len(input_shape) == 4: input_shape = input_shape[1:]

        self.n_bbox = np.zeros(2, dtype = np.int32)
        self.n_stride = np.zeros(2, dtype = np.int32)
        self.n_bbox[1] = (input_shape[1] - EMPTY_AREA[1]) // BOARD_SIZE
        self.n_bbox[0] = (input_shape[0] - EMPTY_AREA[0]) // BOARD_SIZE
        self.n_stride[1] = self.n_bbox[1] // 2
        self.n_stride[0] = self.n_bbox[0] // 2

        self.bbox.assign(self.n_bbox)
        self.stride.assign(self.n_stride)

        print("bbox: {}, stride: {}".format(self.n_bbox, self.n_stride))
        self.built = True

    @tf.function
    def call(self, inputs, *args, **kwargs):
        input_shape = inputs.get_shape()
        print("I've been called with ", input_shape)
        if len(inputs.shape) != 4:
            raise ValueError("Input shape must be equal to 4")

        sizes = [1, self.n_bbox[1], self.n_bbox[0], 1]
        strides = [1, self.n_stride[1], self.n_stride[0], 1]
        print("sizes, strides: ", sizes, strides)

        self.img_list = tf.image.extract_patches(inputs,
                                                 sizes=sizes,
                                                 strides=strides,
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')

        self.img_list = tf.reshape(self.img_list, (-1, self.n_bbox[1], self.n_bbox[0], 3))
        print("Images tensor: ", self.img_list.shape)

        return self.img_list

    def compute_output_shape(self, input_shape):
        if self.img_list is not None:
            return self.img_list.shape
        else:
            sizes = [1, self.n_bbox[1], self.n_bbox[0], 1]
            strides = [1, self.n_stride[1], self.n_stride[0], 1]
            return tf.math.matmul(sizes, strides)

class SampleBoardItemClassifier(BoardItemClassifier):

    def __init__(self, model_dir, img_dir, log_dir = None):
        BoardItemClassifier.__init__(self,
            model_dir=model_dir, img_dir=img_dir, img_size=None, log_dir=log_dir)
        self.train_dataset = None
        self.class_names = ['border', 'edge']

    def load(self):
        """Load a model from directory"""
        print("==> Loading model")
        sample_model = tf.keras.models.load_model(self.model_dir)
        self.model = tf.keras.models.Sequential([BoardBoxesExtractorLayerV3()])
        for layer in sample_model.layers:
            self.model.add(layer)

    def get_model_layers(self):
        layers = super(SampleBoardItemClassifier, self).get_model_layers()
        layers.insert(0, BoardBoxesExtractorLayerV3())
        return layers

    def map_fn(self, path):
        """Upload an image fo given path with specified label - internal"""
        print('Loading file ', path)
        image = tf.image.decode_png(tf.io.read_file(path))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    def get_sample_files(self, num_samples = 1):
        img_dir = Path(self.img_dir)
        file_names = [str(f.absolute()) for f in img_dir.glob('*.png')]
        file_names.extend([str(f.absolute()) for f in img_dir.glob('*.jpg')])
        if len(file_names) < num_samples:
            raise Exception('Insufficient number of images in {} ({} required, {} found)'.format(
                            self.img_dir, num_samples, len(file_names)))
        file_names = file_names[0:num_samples]
        file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
        return file_names

    def init_datasets(self, num_samples = 1, display_samples = False):
        file_names = self.get_sample_files(num_samples)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(file_names)
        self.train_dataset = self.train_dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)
        self.train_dataset = self.train_dataset.shuffle(100)

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


    def display_predictions(self):
        """Display predictions"""
        if self.predictions is None:
            return

        fig = plt.figure(figsize=(8, 8))
        pred_iter = iter(self.predictions)

        bbx = BoardBoxesExtractorLayerV2()
        num_rows = 10
        n_elem = 1

        for _, elements in self.predict_dataset.enumerate():
            images = bbx(elements)
            images = tf.random.shuffle(images)

            for image in images:
                try:
                    prediction = next(pred_iter)
                    pred_label = int(np.argmax(prediction))
                    if prediction[pred_label] < CONFIDENCE_LEVEL:
                        pred_label = -1
                except StopIteration:
                    break

                fig.add_subplot(num_rows, DISPLAY_COLS, n_elem )
                plt.xticks([])
                plt.yticks([])

                plt.imshow(image, cmap=plt.cm.binary)

                plt.title('{}'.format(
                    self.class_names[pred_label] if pred_label >= 0 else 'none'))

                n_elem += 1
                if n_elem >= DISPLAY_COLS * num_rows:
                    break

        plt.tight_layout()
        plt.show()


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
##def get_sample_files(img_dir):
##    """Retrieve specified number of sample files from stone images dataset"""
##    img_dir = Path(img_dir)
##    file_names = [str(f.absolute()) for f in img_dir.glob('*.png')]
##    file_names.extend([str(f.absolute()) for f in img_dir.glob('*.jpg')])
##    file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
##    return file_names
##
##def map_fn(path):
##    image = tf.image.decode_image(tf.io.read_file(path))
##    image = tf.image.convert_image_dtype(image, tf.float32)
##    image = tf.expand_dims(image, axis=0)
##    return image
##
##file_names = get_sample_files('../')
##dataset = tf.data.Dataset.from_tensor_slices(file_names)
##dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
##
##for _, x in dataset.enumerate():
##    bb = BoardBoxesExtractorLayerV3()
##
##    y = bb(x)
##
##    print(y.shape)
##    display_images(y)
##
##
