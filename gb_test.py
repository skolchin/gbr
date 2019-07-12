import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import string as ss

# Constants
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0,0, 0)
DEF_IMG_SIZE = (500, 500)
DEF_IMG_COLOR = (80, 145, 210)

# Global parameters
_DEBUG_ = False
HC_MINDIST = 1
HC_PARAM2 = 10
HC_MAXRADIUS = 20
CANNY_MINVAL = 50
CANNY_MAXVAL = 100
CANNY_APERTURE = 3
HL_RHO = 1
HL_THETA = np.pi/180
HL_THRESHOLD = 100
STONES_THRESHOLD = (10, 200)
STONES_MAXVAL = (255, 255)
STONES_ITER = (4, 2)
BOARD_SIZE = 19

#
# Board recognition functions
#

# Find stones on a board
def find_stones(img, n_thresh, n_maxval, n_iter, f_inv):
    ret, thresh = cv2.threshold(img, n_thresh, n_maxval, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)

    if (f_inv):
        thresh = cv2.bitwise_not(thresh)

    if (_DEBUG_):
       show("Thresholded", thresh)

    stones_img = cv2.dilate(thresh, kernel, iterations=n_iter)

    if (_DEBUG_):
       show("Stones image", stones_img)

    stones = cv2.HoughCircles(stones_img, cv2.HOUGH_GRADIENT, 1,
                                          minDist = HC_MINDIST,
                                          param1 = n_maxval,
                                          param2 = HC_PARAM2,
                                          maxRadius = HC_MAXRADIUS)

    if (stones is None):
       return None
    else:
         print("Stones:", stones.shape)
         return stones[0]

# Find board edges
def find_board_edges(img):
    edges = cv2.Canny(img,CANNY_MINVAL,CANNY_MAXVAL,apertureSize = CANNY_APERTURE)
    lines = cv2.HoughLinesP(edges,HL_RHO,HL_THETA,HL_THRESHOLD)

    lines_img = show_lines("Lines", img.shape, lines, False)
    lines_img = cv2.bitwise_not(lines_img)
    kernel = np.ones((3,3),np.uint8)
    lines_img = cv2.dilate(lines_img, kernel, iterations=1)
    #show("Lines", lines_img)

    lines = cv2.HoughLinesP(lines_img,HL_RHO,HL_THETA,HL_THRESHOLD)
    print(lines.shape)

    xmin = -1
    xmax = -1
    ymin = -1
    ymax = -1

    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]

        if (abs(x1 - x2) < 5 and abs(y1 - y2) < 5):
           if (x1 < xmin or xmin == -1):
              xmin = x1
           if (y1 < ymin or ymin == -1):
              ymin = y1
           if (x2 > xmax or xmax == -1):
              xmax = x2
           if (y2 > ymax or ymax == -1):
              ymax = y2

    p = ((xmin, ymin), (xmax, ymax))
    return p

# Converts X,Y to stone positions
# Returns an array containg stones positions
# Board coordinates are stores as first two array items, board position - as 3,4
def convert_xy(coord, edges, size):
    if (coord is None):
       return np.empty([0,0,0,0])
    else:
         # Make up an empty array
        stones = np.zeros((len(coord), 4), dtype = np.uint16)

        # Calculate distance of board net
        space_x = (edges[1][0] - edges[0][0]) / (size - 1)
        space_y = (edges[1][1] - edges[0][1]) / (size - 1)
        print("Spaces:", space_x, space_y)

        # Loop through, converting board coordinates to integer positions
        for i in range(len(coord)):
            x = coord[i,0] - edges[0][0]
            y = coord[i,1] - edges[0][1]

            stones[i,0] = coord[i,0]
            stones[i,1] = coord[i,1]
            stones[i,2] = int(round(x / space_x, 0)) + 1
            stones[i,3] = int(round(y / space_y, 0)) + 1

            print(i, ":", (stones[i,0], stones[i,1]), "->", stones[i,2], stones[i,3])

        # Remove duplicates
        stones_u = unique_rows(stones)
        return stones

# Find a stone for given coordinates
def find_coord(x, y, coord):
    for i in coord:
        min_x = i[0] - 7
        min_y = i[1] - 7
        max_x = i[0] + 7
        max_y = i[1] + 7
        if (x >= min_x and x <= max_x and y >= min_y and y <= max_y):
           return i

    return (-1, -1, -1, -1)

# Resize an image to minimium allowed size
def preprocess_img(img):
    # Check image size and resize if needed
##    print("Image size:", img.shape)
##    MIN_SIZE = 400
##    f_resize = (img.shape[0] < MIN_SIZE or img.shape[1] < MIN_SIZE)
##    if (f_resize):
##       img = cv2.resize(img,
##           (int(float(MIN_SIZE) / img.shape[1] * img.shape[0]),
##           int(float(MIN_SIZE) / img.shape[0] * img.shape[1])),
##           interpolation = cv2.INTER_CUBIC)
##       kernel = np.array([[-1,-1,-1],
##                   [-1, 9,-1],
##                   [-1,-1,-1]])
##       img = cv2.filter2D(img, -1, kernel)
##       if (_DEBUG_):
##          show('After resizing', img)
    return img

# Function to process an image
def process_img(img):
    # Preprocess an image
    img = preprocess_img(img)

    # Graying out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (_DEBUG_):
       show('Gray image', gray)

    # Find board edges
    board_edges = find_board_edges(gray)

    # Find black stones
    black_stones_xy = find_stones(gray, STONES_THRESHOLD[0], STONES_MAXVAL[0], STONES_ITER[0], False)

    # Find white stones
    white_stones_xy = find_stones(gray, STONES_THRESHOLD[1], STONES_MAXVAL[1], STONES_ITER[1], True)

    # Convert X-Y coordinates to stone positions
    boardSize = BOARD_SIZE
    black_stones = convert_xy(black_stones_xy, board_edges, boardSize)
    white_stones = convert_xy(white_stones_xy, board_edges, boardSize)

    return img, black_stones, white_stones, board_edges, boardSize

#
# Callbacks
#
# Callback for mouse events
def mouse_callback(event):
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))
    f = "Black"
    p = find_coord(x, y, black_stones)
    if (p[0] == -1):
       f = "White"
       p = find_coord(x, y, white_stones)
    if (p[0] >= 0):
       ct = "{f} {a}{b} at ({x},{y})".format(
          f = f,
          a = ss.ascii_uppercase[p[2]-1],
          b = p[3],
          x = round(p[0],0),
          y = round(p[1],0))
       print(ct)
       stone_info.set(ct)

# Load image button callback
def load_img_callback():
    fn = filedialog.askopenfilename(title = "Select file",
       filetypes = (("JPEG files","*.jpg"),("PNG files","*.png"),("all files","*.*")))
    if (fn != ""):
       img = cv2.imread(fn)
       img, black_stones, white_stones, board_edges, boardSize = process_img(img)

       imgtk = img_to_imgtk(img)
       imgPanel.configure(image = imgtk)
       imgPanel.image = imgtk

       board_info.set("Black stones: {}\nWhite stones: {}".format(black_stones.shape[0], white_stones.shape[0]))

# Save SGF button callback
def save_sgf_callback():
    print("Save SGF")

#
# Misc functions
#

# Show image
# Simple wrapper on cv2.imshow
def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()

# Show stones
# The function takes image shape and array of stone coordinates (X,Y,R)
# The function creates a new image with the same shape and draw the stones there
# If f_show = TRUE, image is been shown
def show_stones(title, shape, points, f_show = True):
    img = np.full(shape, COLOR_WHITE[0], dtype=np.uint8)
    for i in points:
        x1 = i[0]
        y1 = i[1]
        r = 7
        print("(", x1, ",", y1, ",", r, ")")
        cv2.circle(img,(x1,y1),r,COLOR_BLACK,-1)

    if (f_show):
       show(title, img)

    return img

# Show lines
# The function takes image shape and array of line coordinates (X1,Y1,X2,Y2)
# The function creates a new image with the same shape and draw the lines there
# If f_show = TRUE, image is been shown
def show_lines(title, shape, lines, f_show = True):
    img = np.full(shape, COLOR_WHITE[0], dtype=np.uint8)
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]

        cv2.line(img,(x1,y1),(x2,y2),COLOR_BLACK,1)

    if (f_show):
       show(title, img)

    return img

# Convert CV2 image to Tkinter format
def img_to_imgtk(img):
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    return imgtk

# Utility function - elmininates duplicates in an array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

# Creates a board image with default parameters
def default_board():
    img = np.zeros((DEF_IMG_SIZE[0], DEF_IMG_SIZE[1], 3), dtype=np.uint8)
    img[:] = DEF_IMG_COLOR

    edges = ((14,14),(DEF_IMG_SIZE[0]-14, DEF_IMG_SIZE[1]-14))
    space_x = (edges[1][0] - edges[0][0]) / (BOARD_SIZE - 1)
    space_y = (edges[1][1] - edges[0][1]) / (BOARD_SIZE - 1)

    for i in range(BOARD_SIZE):
        x1 = int(edges[0][0] + (i * space_x))
        y1 = int(edges[0][1])
        x2 = x1
        y2 = int(edges[1][1])
        cv2.line(img,(x1,y1),(x2,y2),COLOR_BLACK,1)

    for i in range(BOARD_SIZE):
        x1 = int(edges[0][0])
        y1 = int(edges[0][1] + (i * space_y))
        x2 = int(edges[1][0])
        y2 = y1
        cv2.line(img,(x1,y1),(x2,y2),COLOR_BLACK,1)

    return img


#
# Main function
#
# Show initial image
window = tk.Tk()
window.title("Go board")

# Load predifined image
img = cv2.imread("C:\\Users\\skolchin\\Documents\\kol\\gb\\go_board_1.png")
#img, black_stones, white_stones, board_edges, boardSize = process_img(img)
img = default_board()

# Image panel
imgtk = img_to_imgtk(img)
imgPanel = tk.Label(window, image = imgtk)
imgPanel.grid(rowspan = 2, column = 0)

# Info panel
board_info = tk.StringVar()
board_info.set("No stones found")
panel = tk.Label(window, textvariable = board_info)
panel.grid(row = 0, column = 1, columnspan = 2, sticky = tk.E + tk.W)

panel = tk.Button(window, text = "Load image", command = load_img_callback)
panel.grid(row = 1, column = 1, sticky = tk.N + tk.W, padx = 5)

panel = tk.Button(window, text = "Save SGF", command = save_sgf_callback)
panel.grid(row = 1, column = 2, sticky = tk.N + tk.W, padx = 5)

stone_info = tk.StringVar()
stone_info.set("")
panel = tk.Label(window, textvariable = stone_info)
panel.grid(row = 2, columnspan = 2, sticky = tk.W)

# Mouse callback
window.bind('<Button-1>', mouse_callback)

# Display window
window.mainloop()

# This is the end
cv2.destroyAllWindows()
