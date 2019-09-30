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
        self.max_size = 700

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)

        # Image panel and mask
        self.imgPanel = addImagePanel(self.imgFrame,
              caption = "Image",
              btn_params = [["area", True, self.set_area_callback, "Set board area"],
                            ["reset", False, self.transf_reset_callback, "Reset image"],
                            ['edge', False, self.transform_callback, "Transform image"]],
              image = self.src_img,
              max_size = self.max_size,
              scrollbars = False,
              use_mask = True,
              show_mask = True,
              allow_change = True,
              frame_callback = self.update_callback,
              mask_callback = self.mask_callback)

        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        self.imgPanel.buttons['reset'].disabled = True
        self.mask_callback(self.imgPanel.image_mask)

        #self.binder = NBinder()
        #self.binder.bind(self.imgPanel.canvas, '<Button-1>', self.update_callback)

        # Image transformer
        self.transform = None


    def set_area_callback(self, event, tag, state):
        if state:
            self.imgPanel.image_mask.random_mask()
            self.mask_callback(self.imgPanel.image_mask)
            self.imgPanel.image_mask.show()
        else:
            self.imgPanel.image_mask.hide()
        return True

    def mask_callback(self, mask):
        self.imgPanel.caption = "Mask {}".format(mask.scaled_mask)

    def transform_callback(self, event, tag, state):
        if not self.transform is None:
           self.transform.cancel()
           self.transform = None
        else:
            self.imgPanel.image_mask.hide()
            self.transform = ImageTransform(self.imgPanel.canvas, self.imgPanel.image, self.end_transform_callback)
            self.transform.show_coord = True
            self.transform.start()
        return True

    def end_transform_callback(self, t, state):
        if not t.transformed is None:
           self.imgPanel.image = t.transformed
           self.imgPanel.buttons['reset'].disabled = False

        self.imgPanel.buttons['edge'].state = False
        self.transform = None

    def transf_reset_callback(self, event, tag, state):
        self.imgPanel.image_mask.hide()
        self.imgPanel.image = self.src_img
        return False


    def update_callback(self, event):
        print('Update_callback()')


# Main function
def main():

    img = cv2.imread('img\\go_board_47.jpg')
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

