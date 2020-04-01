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
import numpy as np
import cv2
from PIL import Image, ImageTk
import logging
import random
import tkinter as tk

from gr.board import GrBoard
from gr.utils import img_to_imgtk, resize2, format_stone_pos
from gr.grdef import *
from gr.ui_extra import *

# Test dialog
class GrTestDlg(GrDialog):
    pass

# Board edit dialog class
class GrBoardEdit(object):

    def __init__(self, root, img, board_size = 19):
        self.root = root
        self.board_size = board_size

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)

        # Image panel
        self.imgPanel = addImagePanel(self.imgFrame,
              caption = "Image",
              image = img,
              max_size = 600,
              mode = "fit",
              min_size = 300,
              scrollbars = False)
        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        ImgButton(self.imgPanel.headerPanel, tag = "save",
            tooltip = "Testing", command = self.test_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "reset",
            tooltip = "Reset after transformation", command = self.transf_reset_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "area",
            tooltip = "Set board area", command = self.set_area_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "edge",
            tooltip = "Transform image", command = self.transform_callback).pack(side = tk.RIGHT)

        # Image mask
        self.timer_id = None
        self.imgMask = ImageMask(self.imgPanel,
            allow_change = True,
            show_mask = True,
            mode = 'area',
            size = 21,
            mask_callback = self.mask_callback)
        self.mask_callback(self.imgMask)

        # Image transform
        self.imgTransform = ImageTransform(self.imgPanel,
            callback = self.end_transform_callback,
            inplace = False,
            show_coord = True,
            keep = True,
            connect = True)

    def set_area_callback(self, event):
        if event.state:
            self.imgTransform.cancel()
            self.imgMask.default_mask()
            self.mask_callback(self.imgMask)
            self.imgMask.show()
        else:
            self.imgMask.hide()

    def mask_callback(self, mask):
        self.imgPanel.caption = "{} {}".format(mask.mode, mask.scaled_mask)

    def transform_callback(self, event):
        if not event.state:
           self.imgTransform.cancel()
        else:
            self.imgMask.hide()
            if self.imgTransform.transform_rect is None:
                self.imgTransform.start()
            else:
                self.imgTransform.show()
                if self.timer_id is not None:
                    self.imgPanel.after_cancel(self.timer_id)
                self.timer_id = self.imgPanel.after(5000, lambda: self.imgTransform.hide())
                event.cancel = True

    def end_transform_callback(self, t, img):
        self.imgPanel.buttons['edge'].state = False
        if img is not None:
            self.imgPanel.caption = "Transf rect {}".format(t.scaled_rect)
            cv2.imshow('Transformed', img)
            if self.timer_id is not None:
                self.imgPanel.after_cancel(self.timer_id)
            self.timer_id = self.imgPanel.after(5000, lambda: self.imgTransform.hide())

    def transf_reset_callback(self, event):
        self.imgMask.hide()
        self.imgTransform.reset()
        event.cancel = True

    def zoom_in_callback(self, event):
        if self.imgPanel.scale[0] > 1.7: return
        self.imgPanel.scale = [x * 1.10 for x in self.imgPanel.scale]
        self.imgPanel.caption = "scale {:.2f}".format(self.imgPanel.scale[0])
        event.cancel = True

    def zoom_out_callback(self, event):
        if self.imgPanel.scale[0] < 0.3: return
        self.imgPanel.scale = [x * 0.90 for x in self.imgPanel.scale]
        self.imgPanel.caption = "scale {:.2f}".format(self.imgPanel.scale[0])
        event.cancel = True

    def move_callback(self, event):
        print(event.x, event.y)

    def test_callback(self, event):
        event.cancel = True


# Main function
def main():

    img = cv2.imread('img\\go_board_1.png')
    #img = cv2.imread('img\\go_board_47.jpg')
    #img = cv2.imread('img\\go_board_15_large.jpg')
    #img = cv2.imread('img\\go_board_8.png')
    if img is None:
        raise Exception('File not found')

    window = tk.Tk()
    window.title("Edit board")
    gui = GrBoardEdit(window, img)
    window.mainloop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

