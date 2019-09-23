#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Image correction dialog
#
# Author:      kol
#
# Created:     23.09.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from gr.board import GrBoard
from gr.utils import img_to_imgtk, resize2, format_stone_pos
from gr.grdef import *
from gr.ui_extra import *
from gr.grlog import GrLog

import numpy as np
import cv2
import sys
import os
from PIL import Image, ImageTk
import logging
import random

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog  as filedialog
else:
    import tkinter as tk
    from tkinter import filedialog


def is_on(a, b, c):
    "Return true iff point c intersects the line segment from a to b."

    def collinear(a, b, c):
        "Return true iff a, b, and c all lie on the same line."
        return (b[0] - a[0]) * (c[1] - a[1]) == (c[0] - a[0]) * (b[1] - a[1])

    def within(p, q, r):
        "Return true iff q is between p and r (inclusive)."
        return p <= q <= r or r <= q <= p

    # (or the degenerate case that all 3 points are coincident)
    return (collinear(a, b, c)
            and (within(a[0], c[0], b[0]) if a[0] != b[0] else
                 within(a[1], c[1], b[1])))


def is_on_w(a,b,c):
    for i in range(3):
        x = c[0] + i - 1
        for j in range(3):
            y = c[1] + j - 1
            if is_on(a, b, (x, y)): return True
    return False

# Board edit dialog class
class GrBoardEdit(object):

    def __init__(self, root, img, board_size = 19):
        self.root = root
        self.src_img = img
        self.board_size = board_size

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.configFrame = tk.Frame(self.root, bd = 1, relief = tk.RAISED)
        self.configFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)
        #self.buttonFrame = tk.Frame(self.root, width = max_size + 10, height = 70, bd = 1, relief = tk.RAISED)
        #self.buttonFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)

        # Board image
        self.boardImgTk = img_to_imgtk(self.src_img)

        # Image panel
        _, self.imgPanel, _ = addImagePanel(self.imgFrame,"Image",
              [["box", False, self.set_edges_callback, "Edges/spacing"],
              ["edge", False, self.transform_callback, "Rotate/skew"]],
           None, None)

        # Canvas and image on canvas
        self.mask_area = None
        self.mask_rect = None
        self.canvas = tk.Canvas(self.imgPanel,
              width = self.src_img.shape[1],
              height = self.src_img.shape[0])
        self.canvasImg = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.boardImgTk)
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Mouse move handler
        self.last_cursor = None
        self.drag_side = None
        self.canvas.bind("<Motion>", self.motion_callback)
        self.canvas.bind('<B1-Motion>', self.drag_callback)
        self.canvas.bind('<B1-ButtonRelease>', self.end_drag_callback)

        # Editors and buttons
        self.editorFrame = tk.Frame(self.configFrame)
        self.editorFrame.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.sizeVar, self.sizeEntry = addField(self.editorFrame, "e", "Board size", 0, 0, self.board_size)

        self.updateBtn = tk.Button(self.configFrame, text = "Update",
                                                    command = self.update_callback)
        self.updateBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)


    def set_edges_callback(self, event, tag, state):
        if state:
            mask = [random.randint(0,200),
                    random.randint(0,200),
                    random.randint(201,self.src_img.shape[1]),
                    random.randint(201,self.src_img.shape[0])]
            self.draw_mask(mask)
        else:
            self.clear_mask()
        return True

    def transform_callback(self, event, tag, state):
        return False

    def update_callback(self):
        pass

    def get_mask_rect_side(self, x, y):
        if self.mask_rect is None:
           return None

        p = (self.canvas.canvasx(x), self.canvas.canvasy(y))
        b = self.canvas.coords(self.mask_rect)
        #print('Move: ', p, '->>', b)

        side = None
        if is_on_w((b[0], b[1]), (b[0], b[3]), p):
            side = 0
        elif is_on_w((b[0], b[1]), (b[2], b[1]), p):
            side = 1
        elif is_on_w((b[2], b[1]), (b[2], b[3]), p):
            side = 2
        elif is_on_w((b[0], b[3]), (b[2], b[3]), p):
            side = 3
        #print('Move: side', side)
        return side

    def motion_callback(self, event):
        CURSORS = ["left_side", "top_side", "right_side", "bottom_side"]
        c = None
        if not self.mask_rect is None:
            side = self.get_mask_rect_side(event.x, event.y)
            if not side is None: c = CURSORS[side]

        if c is None and not self.last_cursor is None:
            # Left rectangle, set cursor to default
            self.canvas.config(cursor='')
            self.last_cursor = None
        elif not c is None and self.last_cursor != c:
            # On a line, set a cursor
            self.canvas.config(cursor=c)
            self.last_cursor = c

    def drag_callback(self, event):
        if self.drag_side is None:
           self.drag_side = self.get_mask_rect_side(event.x, event.y)
           #print('Drag: side', self.drag_side)
        if not self.drag_side is None:
           p = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
           b = list(self.canvas.coords(self.mask_rect))
           if self.drag_side == 0:
                b[0] = max(p[0], 5)
           elif self.drag_side == 1:
                b[1] = max(p[1], 5)
           elif self.drag_side == 2:
                b[2] = min(p[0], self.src_img.shape[1]-5)
           elif self.drag_side == 3:
                b[3] = min(p[1], self.src_img.shape[0]-5)
           #print('Drag:', p, '->', b)
           self.canvas.coords(self.mask_rect, b[0], b[1], b[2], b[3])
           self.draw_mask_shading(b)

    def end_drag_callback(self, event):
        #print('Drag end')
        self.drag_side = None

    def draw_mask(self, mask):
        """Draw a mask for given coordinates. mask is a list of x0,y0,x1,y1"""
        self.draw_mask_shading(mask)
        self.draw_mask_rect(mask)

    def clear_mask(self):
        if not self.mask_area is None:
            for m in self.mask_area:
                self.canvas.delete(m)
            self.mask_area = None
        if not self.mask_rect is None:
            self.canvas.delete(self.mask_rect)
            self.mask_rect = None

    def draw_mask_shading(self, mask):
        """Draw a shading part of mask"""
        def _rect(points):
            return self.canvas.create_polygon(
                  *points,
                  outline = "",
                  fill = "gray",
                  stipple = "gray50")

        # Clean up
        if not self.mask_area is None:
            for m in self.mask_area:
                self.canvas.delete(m)
            self.mask_area = None

        # Create mask points array
        ix = self.src_img.shape[1]
        iy = self.src_img.shape[0]
        mx = mask[0]
        my = mask[1]
        wx = mask[2]
        wy = mask[3]
        self.mask_area = [
          _rect([0, 0, ix, 0, ix, my, 0, my, 0, 0]),
          _rect([0, my, mx, my, mx, iy, 0, iy, 0, my]),
          _rect([mx, wy, ix, wy, ix, iy, mx, iy, mx, wy]),
          _rect([wx, my, ix, my, ix, wy, wx, wy, wx, my])
        ]

    def draw_mask_rect(self, mask):
        """Draws a transparent mask part"""
        # Clean up
        if not self.mask_rect is None:
            self.canvas.delete(self.mask_rect)
            self.mask_rect = None

        # Draw rect
        mx = mask[0]
        my = mask[1]
        wx = mask[2]
        wy = mask[3]
        self.mask_rect = self.canvas.create_rectangle(
          mx, my,
          wx, wy,
          outline = "red",
          width = 2
        )

# Main function
def main():

    img = cv2.imread('img\\go_board_25.png')
    if img is None:
        raise Exception('File not found')

    # Construct interface
    window = tk.Tk()
    window.title("Edit board")

    log = GrLog.init()
    gui = GrBoardEdit(window, img)

    # Main loop
    window.mainloop()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

