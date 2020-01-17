#-------------------------------------------------------------------------------
# Name:        Simple stone classification TF model
# Purpose:     Learn TensorFlow 2.0
#
# Author:      kol
#
# Created:     13.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randrange

IMG_HEIGHT = 20
IMG_WIDTH = 20
NUM_EPOCHS = 20
BATCH_SIZE = 32
DISPLAY_COLS = 6
CONFIDENCE_LEVEL = 0.8

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Board elements classifier model wrapper
class BoardItemClassifier:
    """This class wraps around TF model
    A stone images dataset made by cc/cc_gen.py is required for model training and prediction
    """
    def __init__(self, model_dir, img_dir, img_size = (IMG_WIDTH, IMG_HEIGHT), log_dir = None):
        """Constructor.
        Parameters:
            model_dir   Directory where a model is saved
            img_dir     Root directory of stone images dataset
            img_size    Target image size
        """
        self.model = None
        self.model_dir, self.img_dir, self.img_size, self.log_dir = model_dir, img_dir, img_size, log_dir

        self.image_data_gen = None
        self.train_generator = None
        self.val_generator = None
        self.history = None
        self.predict_generator = None
        self.predict_dataset = None
        self.predictions = None

        self.class_names = np.array([item.name for item in Path(self.img_dir).glob('*') if item.is_dir()])

    def exists(self):
        """Checks saved model presence"""
        return Path(self.model_dir).exists()

    def load(self):
        """Load a model from directory"""
        print("==> Loading model")
        self.model = tf.keras.models.load_model(self.model_dir)

    def build(self):
        """Build new model"""
        print("==> Building model")
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])

    def save(self):
        """Save whole model to specified directory"""
        self.model.save(self.model_dir)

    def init_datasets(self, display_samples = False):
        """Initialize datasets for training"""
        print("==> Loading images")
        self.image_data_gen = ImageDataGenerator(
            rescale=1./255,
            #rotation_range=30,
            #shear_range=30,
            #width_shift_range=.15,
            #height_shift_range=.15,
            #zoom_range=0.5,
            validation_split=0.2)

        self.train_generator = self.image_data_gen.flow_from_directory(
            batch_size=BATCH_SIZE,
            directory=self.img_dir,
            shuffle=True,
            target_size=self.img_size,
            class_mode='sparse',
            subset='training')

        self.val_generator = self.image_data_gen.flow_from_directory(
            batch_size=BATCH_SIZE,
            directory=self.img_dir,
            shuffle=True,
            target_size=self.img_size,
            class_mode='sparse',
            subset='validation')

        if display_samples:
            self.display_sample_images()

    def train(self, epochs = NUM_EPOCHS, display_history = False):
        """Train the model"""
        print("==> Training the model")
        if self.model is None:
            self.build()
        if self.train_generator is None:
            self.init_datasets()

        callbacks = []
        if self.log_dir is not None:
            callbacks.extend([tf.keras.callbacks.TensorBoard(self.log_dir)])

        self.history = self.model.fit_generator(
                  self.train_generator,
                  epochs=epochs,
                  steps_per_epoch=self.train_generator.samples // BATCH_SIZE,
                  validation_data=self.val_generator,
                  validation_steps=self.val_generator.samples // BATCH_SIZE,
                  callbacks = callbacks)

        if display_history:
            self.display_history()

    def predict(self, num_samples = BATCH_SIZE, display_predictions = True):
        """Predict on specified number of samples"""
        if self.model is None:
            raise Exception("Model is empty, either build or load it")

        print("==> Prediction")
        file_names, file_labels = self.get_sample_files(num_samples)
        self.predict_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
        self.predict_dataset = self.predict_dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)
        self.predict_dataset = self.predict_dataset.batch(BATCH_SIZE)

        self.predictions = self.model.predict(self.predict_dataset)

        if display_predictions:
            self.display_predictions()

    def map_fn(self, path, label):
        """Upload an image fo given path with specified label - internal"""
        image = tf.image.decode_png(tf.io.read_file(path))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.img_size)
        return image, label

    def get_sample_files(self, num_samples = BATCH_SIZE):
        """Retrieve specified number of sample files from stone images dataset"""
        file_names = []
        file_labels = []
        for n, d in enumerate(Path(self.img_dir).glob('*')):
            names = [str(f) for f in Path(self.img_dir).joinpath(d).glob('*.png')]
            file_names.extend(names)
            labels = [float(x == d.name) for x in self.class_names]
            file_labels.extend([labels] * len(names))

        random_file_names = []
        random_file_labels = []
        for _ in range(0, num_samples):
            n = randrange(0, len(file_names)-1)
            random_file_names.extend([file_names[n]])
            random_file_labels.extend([file_labels[n]])

        file_names = tf.convert_to_tensor(random_file_names, dtype=tf.string)
        file_labels = tf.convert_to_tensor(random_file_labels)
        file_labels = tf.expand_dims(file_labels, axis=-1)

        return file_names, file_labels


    def display_sample_images(self):
        """Display up to 25 images from training dataset"""
        if self.train_generator is None:
            self.init_datasets()

        images, labels = next(self.train_generator)
        plt.figure(figsize=(5,5))
        for n in range(min(25, images.shape[0])):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(images[n])
            if len(labels.shape) == 1:
                plt.title(self.class_names[int(labels[n])].title())
            else:
                m = np.argmax(labels[n])
                plt.title(self.class_names[int(labels[n, m])].title())
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def display_history(self):
        """Display training history"""
        if self.history is None:
            return

        acc = self.history.history['sparse_categorical_accuracy']
        val_acc = self.history.history['val_sparse_categorical_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = len(acc)
        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def display_predictions(self):
        """Display predictions"""
        if self.predictions is None:
            return

        pred_iter = iter(self.predictions)
        for _, elements in self.predict_dataset.enumerate():
            fig = plt.figure(figsize=(8, 8))
            num_rows = int(np.ceil(elements[0].shape[0] / DISPLAY_COLS))
            n_elem = 1

            for i, e in enumerate(elements[0]):
                g = tf.expand_dims(e, 0)
                x = classifier.model.predict_classes(g)

            for image, labels in zip(elements[0], elements[1]):
                true_label = int(np.argmax(labels))
                try:
                    prediction = next(pred_iter)
                    pred_label = int(np.argmax(prediction))
                    if prediction[pred_label] < CONFIDENCE_LEVEL:
                        pred_label = -1
                except StopIteration:
                    break

                fig.add_subplot(num_rows, DISPLAY_COLS, n_elem)
                plt.xticks([])
                plt.yticks([])
                n_elem += 1

                plt.imshow(image, cmap=plt.cm.binary)

                plt.title('{} ({})'.format(
                    self.class_names[pred_label] if pred_label >= 0 else 'none',
                    self.class_names[true_label]))

            plt.tight_layout()

        plt.show()

