# Go board image recognition (GBR)

This project is aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement.

The project is build on wonderfull [OpenCV](https://opencv.org/) library.

A lot of ideas and algorithms was found on excellent Adrian Rosebrock's [PyImageSearch](https://www.pyimagesearch.com/) site and
thematical [Slashdot](https://stackoverflow.com/questions/tagged/opencv) threads.

The algorithm per se is the following:

1. Detect board properties (board edges, spacing and board size):
    * Transform image using 4-points transformation and set area to be recognized
    * If parameters set - run HoughLinesP to determine line segments, filter out small lines and reconstruct the image. This allows to remove board labels.
    * Run HoughLines to find the lines
    * Separate lines to vertical/horizontal ones
    * Remove duplicates and lines too close to each other
    * Calculate board edges as minimum and maximum coordinates of horizontal/vertical lines
    * Detect a board size as number of horizontal/vertical lines found

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
  * Luminosity equalization using (CLAHE)[http://books.google.com.au/books?hl=en&lr=&id=CCqzMm_-WucC&oi=fnd&pg=PR11&dq=Contrast%20Limited%20Adaptive%20Histogram%20Equalization%20Graphics%20Gems%20IV&ots=mtft15JJbl&sig=acQg6XLt7jzqR0MjO6sYUa0Sjtc#v=onepage&q=Contrast%20Limited%20Adaptive%20Histogram%20Equalization%20Graphics%20Gems%20IV&f=false]
  * Watershed (post-filter).

Filter and board detection parameters can be changed through the interface and saved to property file (.JSON). The property file is loaded automatically when an image is loaded for processing. Board recognition parameters can also be saved in another JSON file with .JGF extension.

Currently, the program performs quite well on a computer-generated boards. In complex cases, additional parameter tuning might be needed.

As for real board images, they have to be manually adjusted to have all edges to be equal to abount 90 degree and board lines - to be horizontal/vertical. After that, they are processed satisfactory, but more tuning on parameters might be required.

Examples of source images and results of their processing:

| Source | Generated |
| ---    | ---       |
| ![1](../master/img/go_board_1.png) | ![1](../master/img/go_board_1_gen.jpg) |
| Plain computer board ([source](https://images.app.goo.gl/qtziTu6xfNFH46o88)) with non-standard number of lines ||
| ![2](../master/img/go_board_13.png) | ![2](../master/img/go_board_13_gen.png) |
| A computer board showing results of score calculation ||
| ![3](../master/img/go_board_8a.png) | ![3](../master/img/go_board_8a_gen.jpg) |
| Real board ([source](https://www.theverge.com/2016/3/8/11178462/google-deepmind-go-challenge-ai-vs-lee-sedol)) stripped from larger image ||
| ![4](../master/img/go_board_47.jpg) | ![3](../master/img/go_board_47_gen.jpg) |
| Real board ([source](https://images.app.goo.gl/tXP2Yp9GBajHgJEr9)) with 4-points transformation and luminosity normalization applied ||

More images are available at [img](../master/img) directory. All of them were either from my parties or found on the Internet.


## Requirements

Python 2.7/3.5, numpy, opencv2

For DLN: Caffe, py-faster-rcnn ([original](https://github.com/rbgirshick/py-faster-rcnn) or any other fork)


## Changelog

02/10/2019

* Image can now be transformed to rectangular suitable for recognigion even if picture was taken from some angle or skewed by using 4-point transformation algorithm implemented in Adrian Rosebrock's [PyImageSearch](https://www.pyimagesearch.com/) imutils package.
* New ImageTransform class to support setting image transformation parameters
* New filter: luminocity equalization, can be applied when different part of an image are exposed differently.


26/09/2019:

* Now it is possible to define exact board area for recognition. This allows to take off board labels and stuff outside the board
* New ImagePanel and ImageMask classses containing all the code to display images and area masks. UI modules adopted to support them.

22/09/2019:

* Line/edges detection completelly rewritten to simplify the code. HoughLinesP detection is now optional and runs only if threshold/minlen params set.

04/09/2019:

* New GrTag module added to support easy image database navigation and usage
* gr.find_board() rewritten to better recognize board edges/net
* Added: logging in gr and grboard modules, log processing and "Show Log" button in user interfaces
* Added: watershed morphing parameter WS_MORPH_(B|W). If set, image is dilated/eroded before applying watershed allowing to separate connected stone circles
* Changed: ViewAnno GUI layout

22/08/2019:

* Changed: gr.gr find_stones() refactored to support adding new filters
* Added: pyramin mean shift filter to smooth complex stone surface

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

- [x] Adopt the algorithm to the photos of real boards

- [x] Make "tagger" interface to simplify image processing

- [ ] Implement board capture from webcam (new interface)

- [ ] Add stones removal/adding/color changing

- [ ] Implement score calculation

- [ ] Add SGF file creation

- [ ] Make a web interface with (probably) cloud deployment

- [x] Make DLN dataset creation and review interfaces (currently - in PASCAL VOC format)

- [ ] Train DLN model to recognize stones on computer boards

- [ ] Train DLN model to recognize stones on real boards

- [ ] Create a mobile app
