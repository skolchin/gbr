# Go board image recognition (GBR)

This is my project aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement.

The project is build on wonderfull [OpenCV](https://opencv.org/) library.

The algorithm per se is the following:

1. Make a gray image from the source image.

1. Detect board pararameters (edges, board size):

  * Run HoughLinesP lines detection which would return multiple line segments

  * Remove lines which are too close to image borders

  * Find board edges (minimum/maximum X and Y of line segments)

  * HoughLines which would return all lines (it returns line orientation but not line origin)

  * Find horizontal and vertical lines and remove lines too close to each other

  * Assume a board size as number of horizontal/vertical lines most close to predefind board sizes

3. Find stones (black and white):

  * Apply pre-filters specified in parameters (channel splitting, thresholding, dilating, eroding, etc)

  * Run HoughCircles to detect circles, convert their X,Y coordinates to board position

  * Apply post-filters to tune circle radius (watershed)


There are some tuning parameters for each of the steps and they are to be adjusted for each particular board. After the tuning, the program performs quite well on a computer-generated boards.

As for real board images, they have to be manually adjusted to have all edges to be equal to abount 90 degree and board lines - to be horizontal/vertical. After that, they are processed satisfactory, but more tuning on parameters might be required.

Examples of source and generated images:

| Source | Generated |
| ---    | ---       |
| ![1](../master/img/go_board_1.png) | ![1](../master/img/go_board_1_gen.jpg) |
| ![2](../master/img/go_board_13.png) | ![2](../master/img/go_board_13_gen.png) |
| ![3](../master/img/go_board_8a.png) | ![3](../master/img/go_board_8a_gen.jpg) |


## Requirements

Python 2.7/3.5, numpy, opencv2. It seems any recent version works fine.

For DLN: Caffe, py-faster-rcnn ([original](https://github.com/rbgirshick/py-faster-rcnn) or any other fork)


## Changelog

03/09/2019:

* New GrTag module added to support easy image database navigation and usage
* gr.find_board() rewritten to better recognize board edges/net
* Added: logging in gr and grboard modules, "Show Log" button in GbrGui()

22/08/2019:

* Changed: gr.gr find_stones() refactored to support adding new filters
* Added: pyrMeanShiftFiltering pre-filter to smooth complex stone surface

19/08/2019:

* Added: waterhed transformation to determine stone radius more precisely

16/08/2019:

* Changed: gr.py, grdef.py, grutils.py finally assembled as a package
* Added: GrBoard class. Code refactored to work with the board class.
* Bugfixes


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

- [x] Add logging during board processing

- [ ] Adopt the algorithm to the photos of real boards (make functions to set edges, rotate board, correct skewness, etc)

- [x] Make "tagger" interface to simplify image processing

- [ ] Implement board capture from webcam (new interface)

- [ ] Add stones removal/adding/color changing

- [ ] Add SGF file creation

- [ ] Make a web interface with (probably) cloud deployment

- [x] Make DLN dataset creation and review interfaces (currently - in PASCAL VOC format)

- [ ] Train DLN model to recognize stones on computer boards

- [ ] Train DLN model to recognize stones on real boards

- [ ] Create a mobile app
