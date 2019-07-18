# Go board image recognition (GBR)

This is my project aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement. 

The project is build on wonderfull OpenCV library.

The algorithm per se is the following:

1. Make a gray image from the source image

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

There are parameters for each of the steps which have to be tuned for each particular board. After the tuning, it performs quite well on a computer-generated boards. However, it couldn't properly process images with extensive glare on the stones.

TODO:

[x] Add stone reconcilation (detection of stones occupying the same position)

[x] Adopt the algorithm to the photos of real boards. This might require image transformations (skew, rotation etc) and background removal.

[x] Add stones removal/adding/color changing

[x] Add SGF file creation

[x] Add deep network to make it universal

[x] Create a mobile app
