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
        self.board_size = board_size

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)

        # Image panel and mask
        self.imgPanel = addImagePanel(self.imgFrame,
              caption = "Image",
              btn_params = [["area", True, self.set_area_callback, "Set board area"],
                            ["reset", False, self.transf_reset_callback, "Reset after transformation"],
                            ['edge', False, self.transform_callback, "Transform image"],
                            ['plus', False, self.zoom_in_callback, "Zoom in"],
                            ['minus', False, self.zoom_out_callback, "Zoom out"]
                            ],
              image = img,
              max_size = 700,
              mode = "clip",
              min_size = 500,
              scrollbars = False)

        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        self.imgPanel.buttons['reset'].disabled = True

        # Image mask
        self.imgMask = ImageMask(self.imgPanel,
            allow_change = True,
            show_mask = True,
            mode = 'grid',
            size = 21,
            mask_callback = self.mask_callback)
        self.mask_callback(self.imgMask)

        # Image transformer
        self.imgTransform = ImageTransform(self.imgPanel,
            callback = self.end_transform_callback)
        self.imgTransform.show_coord = True


    def set_area_callback(self, event, tag, state):
        if state:
            self.imgMask.random_mask()
            self.mask_callback(self.imgMask)
            self.imgMask.show()
        else:
            self.imgMask.hide()
        return True

    def mask_callback(self, mask):
        self.imgPanel.caption = "{} {}".format(mask.mode, mask.scaled_mask)

    def transform_callback(self, event, tag, state):
        if not state and self.imgTransform.started:
           self.imgTransform.cancel()
        else:
            self.imgMask.hide()
            self.imgTransform.start()
        return True

    def end_transform_callback(self, t, state):
        if not state:
           self.imgPanel.buttons['edge'].state = False
        else:
           self.imgPanel.buttons['edge'].state = False
           self.imgPanel.buttons['reset'].disabled = not state
           self.imgPanel.caption = "Transf rect {}".format(t.scaled_rect)

    def transf_reset_callback(self, event, tag, state):
        self.imgPanel.buttons['edge'].state = False
        self.imgMask.hide()
        self.imgTransform.reset()
        self.imgPanel.buttons['reset'].disabled = True
        return False

    def zoom_in_callback(self, event, tag, state):
        if self.imgPanel.scale[0] > 1.7: return
        self.imgPanel.scale = [x * 1.10 for x in self.imgPanel.scale]
        self.imgPanel.caption = "scale {:.2f}".format(self.imgPanel.scale[0])
        return False

    def zoom_out_callback(self, event, tag, state):
        if self.imgPanel.scale[0] < 0.3: return
        self.imgPanel.scale = [x * 0.90 for x in self.imgPanel.scale]
        self.imgPanel.caption = "scale {:.2f}".format(self.imgPanel.scale[0])
        return False


# Main function
def main():

    img = cv2.imread('img\\go_board_1.png')
    #img = cv2.imread('img\\go_board_47.jpg')
    #img = cv2.imread('img\\go_board_15_large.jpg')
    #img = cv2.imread('img\\go_board_8.png')
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

