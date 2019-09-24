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


# Board edit dialog class
class GrBoardEdit(object):

    def __init__(self, root, img, board_size = 19):
        self.root = root
        self.src_img = img
        self.board_size = board_size

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
##        self.configFrame = tk.Frame(self.root, bd = 1, relief = tk.RAISED)
##        self.configFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)
##        self.buttonFrame = tk.Frame(self.root, width = max_size + 10, height = 70, bd = 1, relief = tk.RAISED)
##        self.buttonFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)

        # Board image
        self.boardImgTk = img_to_imgtk(self.src_img)

        # Image panel
        self.imgPanel = addImagePanel(self.imgFrame,
              caption = "Image",
              btn_params = [["edge", False, self.set_edges_callback, "Set board area"]])
        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Canvas and image on canvas
        self.canvas = tk.Canvas(self.imgPanel,
              width = self.src_img.shape[1],
              height = self.src_img.shape[0])
        self.canvasImg = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.boardImgTk)
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Image mask
        self.imageMask = ImageMask(self.canvas, self.src_img.shape, allow_change = True, f_show = False)

##        # Editors and buttons
##        self.editorFrame = tk.Frame(self.configFrame)
##        self.editorFrame.pack(side = tk.LEFT, padx = PADX, pady = PADY)
##
##        self.sizeVar, self.sizeEntry = addField(self.editorFrame, "e", "Board size", 0, 0, self.board_size)
##
##        self.updateBtn = tk.Button(self.configFrame, text = "Update",
##                                                    command = self.update_callback)
##        self.updateBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)


    def set_edges_callback(self, event, tag, state):
        if state:
            self.imageMask.random_mask()
            self.imageMask.show()
        else:
            self.imageMask.hide()
        return True

    def transform_callback(self, event, tag, state):
        return False

    def update_callback(self):
        pass


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

