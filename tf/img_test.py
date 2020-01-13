#-------------------------------------------------------------------------------
# Name:        Image processing with TensorFlow example
# Purpose:     Experiments with processing images in TF 2.0
#
# Author:      kol
#
# Created:     10.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
from functools import partial
from random import randrange
from time import time

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_DIR = str(Path(__file__).absolute().parent.parent.joinpath('cc', 'p'))
MODEL_DIR =  str(Path(__file__).absolute().parent.joinpath('gbr'))
LOGS_DIR = str(Path(__file__).absolute().parent.joinpath('_logs'))
NUM_EPOCHS = 15
BATCH_SIZE = 32
IMG_HEIGHT = 20
IMG_WIDTH = 20
DISPLAY_COLS = 6

def map_fn(path, label):
    image = tf.image.decode_png(tf.io.read_file(path))

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])

    return image, label


def plot_image(img, predicted_label, true_label, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    plt.title('{} ({})'.format(
        class_names[int(np.round(predicted_label, 0))],
        class_names[int(true_label)]))


def show_batch(images, labels, class_names):
    plt.figure(figsize=(5,5))
    for n in range(min(25, images.shape[0])):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(images[n])
        if len(labels.shape) == 1:
            plt.title(class_names[int(labels[n])].title())
        else:
            m = np.argmax(labels[n])
            plt.title(class_names[int(labels[n, m])].title())
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

def plot_predictions(predictions, images, labels, class_names):
    fig = plt.figure(figsize=(8, 8))
    num_rows = int(np.ceil(BATCH_SIZE / DISPLAY_COLS))

    for i in range(BATCH_SIZE-1):
        n = randrange(0, predictions.shape[0]-1)
        fig.add_subplot(num_rows, DISPLAY_COLS, i+1)
        plot_image(images[n], predictions[n], labels[n], class_names)

    plt.tight_layout()
    plt.show()


# Load images
print("==> Loading images")
class_names = np.array([item.name for item in Path(IMG_DIR).glob('*') if item.is_dir()])
print("Classes found: ", class_names)

class_names = []
file_names = []
file_labels = []
for n, d in enumerate(Path(IMG_DIR).glob('*')):
    class_names.extend([d.name])
    names = [str(f) for f in Path(IMG_DIR).joinpath(d).glob('*.png')]
    file_names.extend(names)
    file_labels.extend([1.0 * n] * len(names))

image_count = len(file_names)
print("Images found: ", image_count)

file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
file_labels = tf.convert_to_tensor(file_labels)
file_labels = tf.expand_dims(file_labels, axis=-1)

# Check model exists and load/train it
if Path(MODEL_DIR).exists():
    print("==> Loading pre-trained model")
    model = tf.keras.models.load_model(MODEL_DIR)
else:
    # Generate dataset
    print("==> Generate datasets")
    dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    dataset = dataset.repeat().shuffle(image_count, reshuffle_each_iteration=False)

    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    images, labels = next(iter(dataset))
    show_batch(images.numpy(), labels.numpy(), class_names)

    train_size = int(0.7 * image_count)
    val_size = int(0.15 * image_count)
    test_size = int(0.15 * image_count)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    print("==> Building model")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("==> Training model")
    history = model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              steps_per_epoch=train_size // BATCH_SIZE,
              validation_data=val_dataset,
              validation_steps=val_size // BATCH_SIZE,
              callbacks = [
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=2),
                tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR,
                                               histogram_freq=1,
                                               write_graph=True,
                                               write_grads=True,
                                               write_images=True,
                                               update_freq="batch") ],
              use_multiprocessing=True)

    print("==> Evaluating results")
    model.evaluate(test_dataset, verbose=2)

    print("==> Saving model")
    model.save(MODEL_DIR)

    print("==> Plotting history")
    plot_history(history)

print("==> Generating prediction dataset")
predict_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
predict_dataset = predict_dataset.shuffle(image_count, reshuffle_each_iteration=False)
predict_dataset = predict_dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
predict_dataset = predict_dataset.batch(BATCH_SIZE)
predict_dataset = predict_dataset.take(1)

print("==> Predicting on {} images".format(BATCH_SIZE))
predictions = model.predict(predict_dataset)

print("==> Displaying results")
images, labels = next(iter(predict_dataset))
plot_predictions(predictions, images, labels, class_names)

