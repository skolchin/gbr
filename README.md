# Go board image recognition (GBR)

This is my project aiming to create a program which would be able to analyse a Go board image in order to determine board parameters and stone placement. 

The project is build on wonderfull OpenCV library.

The algorithm per se is the following:

1. Make a gray image from the source image

1. Detect board pararameters (edges, board size):

  2.1. Run HoughLinesP lines detection which would return multiple line segments
  
  2.2. Remove lines which are too close to image borders
  
  2.3. Find board edges (minimum/maximum X and Y of line segments)
  
  2.4. Run HoughLines which would return all lines (it returns line orientation but not line origin)
  
  2.5. Find horizontal and vertical lines and remove lines too close to each other
  
  2.6. Assume a board size as number of horizontal/vertical lines most close to predefind board sizes
  
3. Find stones (black and white):

  3.1. Threshold image to keep only stone-related pixels (for white stone images - also invert it)
  
  3.2. Morph the image (dilate, erode). Add a blur to remove noise
  
  3.3. Run HoughCircles to detect circles which are going to be the stones
  
4. Convert X, Y stone coordinates to board positions

There are parameters for each of the steps which have to be tuned for each particular board. After the tuning, it performs quite well on a computer-generated boards. However, it couldn't properly process images with extensive glare on the stones.

TODO:
1. Add stone reconcilation (detection of stones occupying the same position)

2. Adopt the algorithm to the photos of real boards. This might require image transformations (skew, rotation etc) and background removal.

3. Add stones removal/adding/color changing

4. Add SGF file creation

5. Add deep network to make it universal

6. Create a mobile app
