#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     New GBR UI
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------

from gr.board import GrBoard
from gr.grdef import *
from gr.ui_extra import NBinder, ImgButton, ImagePanel, StatusPanel, ImageMask, ImageTransform
from gr.grlog import GrLog
from gr.utils import format_stone_pos

import numpy as np
import cv2
import sys
from PIL import Image, ImageTk
import logging

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

# GUI class
class GbrGUI2(object):

    # Constructor
    def __init__(self, root):
        self.root = root
        self.buttons = dict()
        self.board = GrBoard()
        self.binder = NBinder()

        self.internalFrame = tk.Frame(self.root)
        self.internalFrame.pack(fill = tk.BOTH, expand = True)

        self.__init_menu()
        self.__init_toolbar()
        self.__init_window()
        self.__init_statusbar()

    # Initialization functions
    def __init_menu(self):
        pass

    def __init_toolbar(self):
        toolbarPanel = tk.Frame(self.internalFrame, bd = 1, relief = tk.RAISED)
        toolbarPanel.pack(side = tk.TOP, fill = tk.X, expand = False)

        self.buttons['open'] = ImgButton(toolbarPanel,
            tag = "open", tooltip = "Open image", callback = self.open_image_callback)

        self.buttons['edge'] = ImgButton(toolbarPanel,
            tag = "edge", tooltip = "Transform image", disabled = True,
            callback = self.transform_callback)

        self.buttons['area'] = ImgButton(toolbarPanel,
            tag = "area", tooltip = "Set board area", disabled = True,
            callback = self.set_area_callback)

        self.buttons['params'] = ImgButton(toolbarPanel,
            tag = "params", tooltip = "Detection params", disabled = True,
            callback = self.set_params_callback)

        self.buttons['detect'] = ImgButton(toolbarPanel,
            tag = "detect", tooltip = "Detect stones", disabled = True,
            callback = self.detect_callback)

        self.buttons['save'] = ImgButton(toolbarPanel,
            tag = "save", tooltip = "Save as SGF", disabled = True,
            callback = self.save_sgf_callback)

        self.buttons['reset'] = ImgButton(toolbarPanel,
            tag = "reset", tooltip = "Reset", disabled = True,
            callback = self.reset_callback)

        for b in self.buttons.keys():
            self.buttons[b].pack(side = tk.LEFT, padx = 2, pady = 2)


    def __init_window(self):
        # Image panel
        img = cv2.imread("ui\\def_board.png")
        self.imagePanel = ImagePanel(self.internalFrame,
            image = img,
            mode = "fit",
            max_size = 500,
            min_size = 300,
            frame_callback = self.mouse_click_callback)
        self.imagePanel.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        # Image mask
        self.imageMask = ImageMask(self.imagePanel,
            allow_change = True,
            show_mask = False,
            mode = 'grid')

        # Image transformer
        self.imageTransform = ImageTransform(self.imagePanel)

        ## Mouse move
        ##self.root.bind('<Motion>', self.mouse_move_callback)


    def __init_statusbar(self):
        self.statusBar = StatusPanel(self.internalFrame,
            callback = self.status_click_callback,
            bd = 1, relief = tk.SUNKEN)

        self.statusBar.pack(side = tk.BOTTOM, fill = tk.X, expand = False)

    #
    # Callbacks
    #
    def open_image_callback(self, event, tag, state):
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "":
            self.load_image(fn)
        return False

    def transform_callback(self, event, tag, state):
        return False

    def set_area_callback(self, event, tag, state):
        if state:
            self.imageMask.show()
        else:
            self.imageMask.hide()
        return True

    def set_params_callback(self, event, tag, state):
        return False

    def detect_callback(self, event, tag, state):
        if not self.board.is_gen_board:
            self.detect_stones()
        return False

    def save_sgf_callback(self, event, tag, state):
        if not self.board.is_gen_board:
            self.detect_stones()
        return False

    def reset_callback(self, event, tag, state):
        return False

    def mouse_move_callback(self, event):
        x,y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(x, y)
        if widget == self.imagePanel.canvas and not self.board.results is None:
            x, y = self.imagePanel.frame2image((event.x, event.y))
            stone, bw = self.board.find_stone(x, y)
            if not stone is None:
                bw = "Black" if bw == "B" else "White"
                self.statusBar.set("{} {}".format(bw, format_stone_pos(stone)))
            else:
                self.statusBar.set("")

    def mouse_click_callback(self, event):
        if not self.board.is_gen_board:
            x, y = self.imagePanel.frame2image((event.x, event.y))
            stone, bw = self.board.find_stone(x, y)
            if not stone is None:
                bw = "Black" if bw == "B" else "White"
                self.statusBar.set("{} {}".format(bw, format_stone_pos(stone)))
            else:
                self.statusBar.set("")

    def status_click_callback(self, event):
        GrLog.show(self.root)

    #
    # Core functions
    #
    def load_image(self, filename):
        GrLog.clear()
        try:
            # Clean up
            self.buttons['area'].state = False
            self.imageMask.hide()

            # Load image
            self.board.load_image(filename, f_process = False, f_with_params = True)

            # Display loaded image and mask
            self.imagePanel.set_image(self.board.image)
            self.imageMask.scaled_mask = self.board.area_mask

            # Reset button states
            self.buttons['reset'].disabled = not self.board.can_reset_image
            for b in ["edge", "area", "params", "detect"]:
                self.buttons[b].disabled = False

            # Update status
            if GrLog.numErrors() > 0:
                self.statusBar.set("Errors during file loading, click here to see the log")
            else:
                self.statusBar.set_file("File loaded", self.board.image_file)

        except:
            logging.exception("Error")
            self.statusBar.set("Errors during file loading, click here to see the log")


    def detect_stones(self):
        GrLog.clear()
        try:
            # Process
            self.board.process()

            # Update status
            if GrLog.numErrors() > 0:
                self.statusBar.set("Errors during processing, click here to see the log")
            else:
                self.statusBar.set("Detected: {} black, {} white".format(
                    len(self.board.black_stones), len(self.board.white_stones)))

        except:
            logging.exception("Error")
            self.statusBar.set("Errors during file loading, click here to see the log")

    #
    # Utility functions
    #

# Main function
def main():
    # Construct interface
    window = tk.Tk()
    window.title("Go board")
    window.minsize(300, 300)

    log = GrLog.init()
    gui = GbrGUI2(window)

    window.mainloop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
