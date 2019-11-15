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

# Options dialog class
class GbrOptionsDlg(tk.Toplevel):
        def __init__(self, parent, *args, **kwargs):
            tk.Toplevel.__init__(self, *args, **kwargs)

            self.minsize(300, 300)
            self.parent = parent
            self.transient(parent.root)
            self.attributes("-toolwindow", True)
            self.title("Parameters")

            x = parent.root.winfo_x() + parent.root.winfo_width()
            y = parent.root.winfo_y() + 50
            self.geometry("%+d%+d" % (x, y))

            internalFrame = tk.Frame(self)
            internalFrame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            self.tkVars = self.add_switches(internalFrame, self.parent.board.params)

            buttonFrame = tk.Frame(self, bd = 1, relief = tk.RAISED)
            buttonFrame.pack(side = tk.BOTTOM, fill = tk.X)

            tk.Button(buttonFrame, text = "Detect",
                command = self.detect_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

            tk.Button(buttonFrame, text = "Close",
                command = self.close_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

            self.focus_set()
            self.resizable(False, False)

        def detect_callback(self):
            p = dict()
            for key in self.tkVars.keys():
                p[key] = self.tkVars[key].get()
            self.parent.board.params = p
            self.parent.detect_stones()

        def close_callback(self):
            self.destroy()

        # Add Scale widgets with board recognition parameters
        def add_switches(self, rootFrame, params, max_in_row = 6):
            n = 1
            ncol = 0
            frame = None
            vars = dict()

            # Add a tabbed notebook
            nb = ttk.Notebook(rootFrame)
            nb.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            # Get unique tabs
            tabs = set([e[2] for e in GR_PARAMS_PROP.values() if e[2]])

            # Add switches to notebook tabs
            for tab in sorted(tabs):
                # Add a tab frame
                nbFrame = tk.Frame(nb, width = 400)
                nb.add(nbFrame, text = tab)
                frame = None
                n = 0
                ncol = 0

                # Iterate through the params processing only ones belonging to current tab
                keys = [key for key in params.keys() if key in GR_PARAMS_PROP and GR_PARAMS_PROP[key][2] == tab]
                for key in sorted(keys, key = lambda k: GR_PARAMS_PROP[k][4] if len(GR_PARAMS_PROP[k]) > 4 else 0):
                    if (n == max_in_row or frame is None):
                        frame = tk.Frame(nbFrame, width = 400)
                        frame.grid(row = 0, column = ncol, padx = 3, pady = 3)
                        n = 0
                        ncol = ncol + 1

                    # Add a switch
                    caption = GR_PARAMS_PROP[key][3] if len(GR_PARAMS_PROP[key]) > 3 else key
                    label = tk.Label(frame, text = caption)
                    label.grid(row = n, column = 0, padx = 2, pady = 0, sticky = "s", ipady=4)

                    v = tk.IntVar()
                    v.set(params[key])
                    scale = tk.Scale(frame, from_ = GR_PARAMS_PROP[key][0],
                                            to = GR_PARAMS_PROP[key][1],
                                            orient = tk.HORIZONTAL,
                                            variable = v)
                    scale.grid(row = n, column = 1, padx = 2, pady = 0)
                    vars[key] = v

                    n = n + 1
            return vars


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
            tag = "area", tooltip = "Define board", disabled = True,
            callback = self.set_grid_callback)

        self.buttons['detect'] = ImgButton(toolbarPanel,
            tag = "detect", tooltip = "Detect stones", disabled = True,
            callback = self.detect_callback)

        self.buttons['params'] = ImgButton(toolbarPanel,
            tag = "params", tooltip = "Detection params", disabled = True,
            callback = self.set_params_callback)

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
        self.imagePanel.pack(side = tk.TOP, fill = tk.BOTH, expand = True,
            padx = 2, pady = 2)

        # Image mask
        self.imageMask = ImageMask(self.imagePanel,
            allow_change = True,
            show_mask = False,
            mode = 'grid',
            mask_callback = self.mask_callback)

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
        """Open button click"""
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "":
            self.load_image(fn)
        return False

    def transform_callback(self, event, tag, state):
        """Transform button click"""
        return False

    def set_grid_callback(self, event, tag, state):
        """Board grid button click"""
        if state:
            self.detect_edges()
            self.imageMask.show()
        else:
            self.imageMask.hide()
        return True

    def set_params_callback(self, event, tag, state):
        """Detection params button click"""
        self.optionsDlg = GbrOptionsDlg(self)
        return False

    def detect_callback(self, event, tag, state):
        """Detect button click"""
        if not self.board.is_gen_board:
            self.detect_stones()
        return False

    def save_sgf_callback(self, event, tag, state):
        """SGF save button click"""
        return False

    def reset_callback(self, event, tag, state):
        """Open button click"""
        return False

##    def mouse_move_callback(self, event):
##        x,y = self.root.winfo_pointerxy()
##        widget = self.root.winfo_containing(x, y)
##        if widget == self.imagePanel.canvas and not self.board.results is None:
##            x, y = self.imagePanel.frame2image((event.x, event.y))
##            stone, bw = self.board.find_stone(x, y)
##            if not stone is None:
##                bw = "Black" if bw == "B" else "White"
##                self.statusBar.set("{} {}".format(bw, format_stone_pos(stone)))
##            else:
##                self.statusBar.set("")

    def mouse_click_callback(self, event):
        """Board image mouse click"""
        if not self.board.is_gen_board:
            x, y = self.imagePanel.frame2image((event.x, event.y))
            stone, bw = self.board.find_stone(x, y)
            if not stone is None:
                bw = "Black" if bw == "B" else "White"
                self.statusBar.set("{} {}".format(bw, format_stone_pos(stone)))
            else:
                self.statusBar.set("")

    def status_click_callback(self, event):
        """Status bar mouse click"""
        GrLog.show(self.root)

    def mask_callback(self, mask):
        """Mask resizing finished"""
        if not self.board.is_gen_board:
           self.board.param_board_edges = mask.scaled_mask

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
            self.imageMask.scaled_mask = self.board.param_board_edges

            # Reset button states
            self.buttons['reset'].disabled = not self.board.can_reset_image
            for b in ["edge", "area", "params", "detect"]:
                self.buttons[b].disabled = False

            # Update status
            if GrLog.numErrors() > 0:
                self.statusBar.set("Errors during file loading, click here for the log")
            else:
                self.statusBar.set_file("File loaded", self.board.image_file)

        except:
            logging.exception("Error")
            self.statusBar.set("Errors during file loading, click here for the log")

    def detect_edges(self, f_force = False):
        if not self.board.param_board_edges is None and not f_force:
           return

        GrLog.clear()
        try:
            self.board.detect_edges()
            if GrLog.numErrors() > 0:
               self.statusBar.set("Automatic board detection failed, click here for the log")
            else:
                self.statusBar.set("Board size detected as {s}x{s}".format(
                                        s = self.board.board_size))
                self.imageMask.scaled_mask = self.board.param_board_edges
                self.imageMask.size = self.board.board_size

        except:
            logging.exception("Error")
            self.statusBar.set("Errors during grid detection, click here for the log")



    def detect_stones(self):
        GrLog.clear()
        try:
            # Process
            self.board.process()

            # Update status
            if GrLog.numErrors() > 0:
                self.statusBar.set("Errors during processing, click here for the log")
            else:
                self.statusBar.set("{b} black, {w} white stones on {s}x{s} board detected".format(
                    b = len(self.board.black_stones),
                    w = len(self.board.white_stones),
                    s = self.board.board_size))

        except:
            logging.exception("Error")
            self.statusBar.set("Errors during processing, click here for the log")

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
