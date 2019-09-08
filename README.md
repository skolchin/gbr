# Go board image recognition (GBR)

This is my project aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement.

The project is build on wonderfull [OpenCV](https://opencv.org/) library.

The algorithm per se is the following:

1. Detect board properties (board edges, spacing and board size):
    * Remove parts of image close to border
    * Run HoughLinesP detection and determine line segments
    * Calculate board edges as minimum and maximum coordinates of horizontal/vertical line segments
    * Make up a new image and draw all line segments (thus removing any noise)
    * Run HoughLines to determine lines (it returns line orientation but not line origin)
    * Assume a board size as number of horizontal/vertical lines found

2. Find stones (black and white):
    * Apply pre-filters with parameters specified through the interface
    * Run HoughCircles to detect circles and convert found X,Y coordinates to board position
    * Apply post-filters to tune stone radius

3. Eliminate duplicates where black and white stones occupy the same board position

Currently, the following filters are implemented:
  * Channel splitting (red channel is used in white stone detections, blue - in black one)
  * Thresholding
  * Dilating
  * Eroding
  * Blur
  * Pyramid mean filtering (useful when stones have textured faces or extensive glare)
  * Watershed (post-filter).

Filter and board detection parameters can be changed through the interface and saved to property file (.JSON). The property file is loaded automatically when an image is loaded for processing. Board recognition parameters can also be saved in another JSON file with .JGF extension.

Currently, the program performs quite well on a computer-generated boards. In complex cases, additional parameter tuning might be needed.

As for real board images, they have to be manually adjusted to have all edges to be equal to abount 90 degree and board lines - to be horizontal/vertical. After that, they are processed satisfactory, but more tuning on parameters might be required.

Examples of source and generated images:

| Source | Generated |
| ---    | ---       |
| ![1](../master/img/go_board_1.png) | ![1](../master/img/go_board_1_gen.jpg) |
| ![2](../master/img/go_board_13.png) | ![2](../master/img/go_board_13_gen.png) |
| ![3](../master/img/go_board_8a.png) | ![3](../master/img/go_board_8a_gen.jpg) |


## Requirements

Python 2.7/3.5, numpy, opencv2, pathlib. It seems any recent version works fine.

For DLN: Caffe, py-faster-rcnn ([original](https://github.com/rbgirshick/py-faster-rcnn) or any other fork)


## Changelog

04/09/2019:

* New GrTag module added to support easy image database navigation and usage
* gr.find_board() rewritten to better recognize board edges/net
* Added: logging in gr and grboard modules, log processing and "Show Log" button in user interfaces
* Added: watershed morphing parameter WS_MORPH_(B|W). If set, image is dilated/eroded before applying watershed allowing to separate connected stone circles
* Changed: ViewAnno GUI layout

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

- [x] Add logging during board processing/dataset generation

- [ ] Adopt the algorithm to the photos of real boards (add functions to set edges, rotate board, correct skewness, etc)

- [x] Make "tagger" interface to simplify image processing

- [ ] Implement board capture from webcam (new interface)

- [ ] Add stones removal/adding/color changing

- [ ] Add SGF file creation

- [ ] Make a web interface with (probably) cloud deployment

- [x] Make DLN dataset creation and review interfaces (currently - in PASCAL VOC format)

- [ ] Train DLN model to recognize stones on computer boards

- [ ] Train DLN model to recognize stones on real boards

- [ ] Create a mobile app
