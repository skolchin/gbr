# Go board image recognition (GBR)

This is my project aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement.

The project is build on wonderfull [OpenCV](https://opencv.org/) library.

The algorithm per se is the following:

1. Make a gray image from the source image and extract a blue channel. Gray image will be used for lines and black stones detection, while blue channel - for white stones detection.

1. Detect board pararameters (edges, board size):

  * Run HoughLinesP lines detection which would return multiple line segments

  * Remove lines which are too close to image borders

  * Find board edges (minimum/maximum X and Y of line segments)

  * HoughLines which would return all lines (it returns line orientation but not line origin)

  * Find horizontal and vertical lines and remove lines too close to each other

  * Assume a board size as number of horizontal/vertical lines most close to predefind board sizes

3. Find stones (black and white):

  * Threshold image to keep only stone-related pixels (for white stone images - also invert it)

  * Morph the image (dilate, erode). Add a blur to remove noise

  * Run HoughCircles to detect circles which are going to be the stones

4. Convert X, Y stone coordinates to board positions

There are some tuning parameters for each of the steps and they are to be adjusted for each particular board. After the tuning, the program performs quite well on a computer-generated boards.

As for real board images, they have to be manually adjusted to have all edges to be equal to abount 90 degree and board lines - to be horizontal/vertical. After that, they are processed satisfactory, but more tuning on parameters might be required.

Examples of source and generated images:

| Source | Generated |
| ---    | ---       |
| ![1](../master/img/go_board_1.png) | ![1](../master/img/go_board_1_gen.jpg) |
| ![2](../master/img/go_board_13.png) | ![2](../master/img/go_board_13_gen.png) |
| ![3](../master/img/go_board_8a.png) | ![3](../master/img/go_board_8a_gen.jpg) |


## Requirements

Python 2.7/3.5, packages: numpy, opencv2, tkinter, pathlib


## Changelog

13/08/2019:

* Added: update_jgf.py script to update all board info files for images where recognition parameters (JSON) exist
* Added: simple stone position reconcilation (white stones precendent)
* Added: function to show detections on generated board

07/08/2019:

* Changed: net/make_dataset.py now creates both test and training DS in PASCAL VOC format
* Changed: net/test_net.py uses Caffe to run network (not OpenCV.dnn)

01/08/2019:

* Added: support for large images
* Added: stone radius saving in JGF file

30/07/2019:

* Added Python 2.7 support


24/07/2019:

* GUI rewritten: all code moved from main() to GbrGUI class

* Extra info on board save

## TODO

- [x] Find some ways to deal with glare on the stones

- [x] Allow to save recognized stone positions

- [x] Add stone reconcilation (detection of stones occupying the same position)

- [ ] Adopt the algorithm to the photos of real boards. This might require image transformations (skew, rotation etc) and background removal.

- [ ] Add stones removal/adding/color changing

- [ ] Add SGF file creation

- [ ] Make a web interface and deploy to cloud

- [ ] Add deep network to make it universal

- [ ] Create a mobile app
