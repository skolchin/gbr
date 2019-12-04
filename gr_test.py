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
from PIL import Image, ImageTk
import logging
import random
import tkinter as tk

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
              max_size = 500,
              mode = "clip",
              min_size = 300,
              scrollbars = False,
              frame_callback = self.frame_callback)
        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        ImgButton(self.imgPanel.headerPanel, tag = "reset",
            tooltip = "Reset after transformation", command = self.transf_reset_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "minus",
            tooltip = "Zoom out", command = self.zoom_out_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "plus",
            tooltip = "Zoom in", command = self.zoom_in_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "area",
            tooltip = "Set board area", command = self.set_area_callback).pack(side = tk.RIGHT)
        ImgButton(self.imgPanel.headerPanel, tag = "edge",
            tooltip = "Transform image", command = self.transform_callback).pack(side = tk.RIGHT)

##              btn_params = [
##                ['params', False, self.test_callback, "Test", GrTestDlg],
##                ['plus', False, self.zoom_in_callback, "Zoom in"],
##                ['minus', False, self.zoom_out_callback, "Zoom out"],
##                ["area", True, self.set_area_callback, "Set board area"],
##                ["reset", False, self.transf_reset_callback, "Reset after transformation"],
##                ['edge', False, self.transform_callback, "Transform image"]
##              ],

        # Image mask
        self.imgMask = ImageMask(self.imgPanel,
            allow_change = True,
            show_mask = True,
            mode = 'area',
            size = 21,
            mask_callback = self.mask_callback)
        self.mask_callback(self.imgMask)

        # Image transform
        self.imgTransform = ImageTransform(self.imgPanel,
            callback = self.end_transform_callback)
        self.imgTransform.show_coord = True

        # Image marker
        self.imgMarker = ImageMarker(self.imgPanel, flash = 3)

        # Button group
        self.bg = ImgButtonGroup(self.imgPanel)
        self.bg.add_group("g1", ["area", "edge"], BG_DEPENDENT)
        self.bg.add_group("g2", ["plus", "area"], BG_INDEPENDENT)

    def set_area_callback(self, event):
        if event.state:
            self.imgTransform.cancel()
            self.imgMask.random_mask()
            self.mask_callback(self.imgMask)
            self.imgMask.show()
        else:
            self.imgMask.hide()

    def mask_callback(self, mask):
        self.imgPanel.caption = "{} {}".format(mask.mode, mask.scaled_mask)

    def transform_callback(self, event):
        if not event.state and self.imgTransform.started:
           self.imgTransform.cancel()
        else:
            self.imgMask.hide()
            self.imgTransform.start()

    def end_transform_callback(self, t, state):
        if not state:
           #self.imgPanel.buttons['edge'].state = False
           pass
        else:
           #self.imgPanel.buttons['edge'].state = False
           #self.imgPanel.buttons['reset'].disabled = not state
           self.imgPanel.caption = "Transf rect {}".format(t.scaled_rect)

    def transf_reset_callback(self, event):
        #self.imgPanel.buttons['edge'].state = False
        self.imgMask.hide()
        self.imgTransform.reset()
        #self.imgPanel.buttons['reset'].disabled = True
        return False

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

    def frame_callback(self, event):
        p = self.imgPanel.frame2image((event.x, event.y))
        self.imgMarker.add_stone([p[0], p[1], 1, 1, 20])
        if not self.imgMarker.is_shown:
            self.imgMarker.show()

    def test_callback(self, event, tag, state):
        return True

##        b = NBinder()
##        b.trigger(self.imgPanel, '<Resize>', ResizeEvent(self.imgPanel, [1.0, 1.0], [1.0, 1.0]))

# Main function
def main():

    #img = cv2.imread('img\\go_board_1.png')
    img = cv2.imread('img\\go_board_47.jpg')
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

