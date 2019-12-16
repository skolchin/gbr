
## Changelog

12/12/2019

* New option: automatic selection of recognition parameters available in Options dialog


04/12/2019

New user-friendly UI created (gbr2.py). Primary features:

* Only one (source) image is displayed
* Separate dialog for recognition parameters, with ability to view how parameter changes affects recognition results
* Automatic parameters update
* Ability to set board edges (area mask), transform skewed and distorted images
* New Stones dialog displaying list of detected stones
* Logging and debug images displaying
* SGF file saving

Old one (gbr.py) put on hold and will no longer be maintained.

13/11/2019

* Added saving to SGF

02/10/2019

* Image can now be transformed to rectangular suitable for recognigion even if picture was taken from some angle or skewed by using 4-point transformation algorithm implemented in imutils package.
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
