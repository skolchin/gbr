#-------------------------------------------------------------------------------
# Name:        MNIST TensorFlow example
# Purpose:     Experiments with TensorFlow
#
# Author:      kol
#
# Created:     09.01.2020
# Copyright:   (c) kol 2020
#-------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from random import randrange

def plot_image(predictions, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions),
                                true_label),
                                color = color)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if Path('./mnist.m').exists():
    print("Loading pre-trained model")
    model = tf.keras.models.load_model('mnist.m')
else:
    print("Training the model")
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.save('./mnist.m')

model.evaluate(x_test,  y_test, verbose=2)
predictions = model.predict(x_test)

max_count = 10
num_rows = 5

fig = plt.figure(figsize=(8,4))

for i in range(max_count):
    n = randrange(0, predictions.shape[0]-1)
    fig.add_subplot(num_rows, max_count / num_rows, i+1)
    plot_image(predictions[n], y_test[n], x_test[n])
    if i >= max_count-1:
        break

plt.tight_layout()
plt.show()
