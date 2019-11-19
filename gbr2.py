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
from gr.ui_extra import NLabel, NBinder, ImgButton, ImagePanel, StatusPanel, \
    ImageMask, ImageTransform, GrDialog
from gr.grlog import GrLog
from gr.utils import format_stone_pos, resize, img_to_imgtk

import numpy as np
import cv2
import os
from PIL import Image, ImageTk
import logging

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox

# Debug info dialog class
class GbrDebugDlg(GrDialog):
    def __init__(self, parent, *args, **kwargs):
        GrDialog.__init__(self, parent, *args, **kwargs)

    def get_minsize(self):
        return (300, 400)

    def get_title(self):
        return "Debug info"

    def get_offset(self):
        return (15, 200)

    def init_frame(self):
        sbr = tk.Scrollbar(self.internalFrame)
        self.canvas = tk.Canvas(self.internalFrame, yscrollcommand=sbr.set)
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        sbr.pack(side = tk.RIGHT, fill = tk.Y)
        sbr.config(command = self.canvas.yview)

        self.debugFrame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0), window = self.debugFrame, anchor='nw')
        self.add_debug_info(self.debugFrame)

        self.canvas.bind('<Configure>', self.on_scroll_configure)

    def init_buttons(self):
        tk.Button(self.buttonFrame, text = "Save images",
            command = self.save_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)
        GrDialog.init_buttons(self)

    def save_click_callback(self):
        """Save button click callback"""
        self.save_debug_info()

    def dbg_img_click_callback(self, event):
        """Mouse click on debug info panel callback"""
        w = event.widget
        k = w.tag
        cv2.imshow(k, self.parent.board.debug_images[k])

    def on_scroll_configure(self, event):
        """Canvas config callback"""
        self.canvas.configure(scrollregion = self.debugFrame.bbox('all'))

    def add_debug_info(self, root):
        """Adds debug information images"""
        shape = self.parent.board.board_shape
        debug_img = self.parent.board.debug_images

        if debug_img is None:
            return

        nrow = 0
        ncol = 0
        sz = min(int(shape[CV_WIDTH] / 2) - 5, 150)

        # Add analysis result images
        for key in sorted(debug_img.keys()):
            frame = tk.Frame(root)
            frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

            img = resize(debug_img[key], sz)

            imgtk = img_to_imgtk(img)
            panel = NLabel(frame, image = imgtk, tag = key)
            panel.image = imgtk
            panel.grid(row = 0, column = 0)
            panel.bind('<Button-1>', self.dbg_img_click_callback)

            panel = tk.Label(frame, text = key)
            panel.grid(row = 1, column = 0, sticky = "nswe")

            ncol = ncol + 1
            if ncol > 1:
                nrow = nrow + 1
                ncol = 0

    def save_debug_info(self):
        """Saves debug images to given directory"""
        debug_img = self.parent.board.debug_images
        if debug_img is None:
            return

        dn = filedialog.askdirectory(initialdir=os.getcwd(),
                                        title='Please select a directory')
        if dn == "":
            return

        n = 0
        for key in debug_img.keys():
            cv2.imwrite(os.path.join(dn, key.lower() + ".png"), debug_img[key])
            n += 1

        messagebox.showinfo("Debug images", "{} files saved to {}".format(n, dn))

# Options dialog class
class GbrOptionsDlg(GrDialog):
    def __init__(self, parent, *args, **kwargs):
        GrDialog.__init__(self, parent, *args, **kwargs)

    def get_title(self):
        return "Parameters"

    def init_params(self, args, kwargs):
        self.board_size_label = None
        self.board_size_scale = None
        self.board_size_disabled = None
        self.debug_dlg = None

    def init_frame(self):
        self.tkVars = self.add_switches(self.internalFrame, self.parent.board.params)

    def init_buttons(self):
        self.btn_image = ImgButton.get_ui_image("detect_flat.png")
        tk.Button(self.buttonFrame, text = "Detect",
            image = self.btn_image, compound="left",
            command = self.detect_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        self.log_button = tk.Button(self.buttonFrame, text = "Log",
            state = tk.DISABLED if GrLog.numErrors() == 0 is None else tk.NORMAL,
            command = self.log_click_callback)
        self.log_button.pack(side = tk.LEFT, padx = 5, pady = 5)

        self.dbg_button = tk.Button(self.buttonFrame, text = "Debug",
            state = tk.DISABLED if self.parent.board.results is None else tk.NORMAL,
            command = self.debug_click_callback)
        self.dbg_button.pack(side = tk.LEFT, padx = 5, pady = 5)

        GrDialog.init_buttons(self)

    def update_controls(self):
        if self.dbg_button is not None:
            self.dbg_button.configure(
                state = tk.DISABLED if self.parent.board.results is None else tk.NORMAL)
        if self.log_button is not None:
            self.log_button.configure(
                state = tk.DISABLED if GrLog.numErrors() == 0 is None else tk.NORMAL)

    def detect_click_callback(self):
        """Detect button click callback"""
        # Save changed params
        p = dict()
        for key in self.tkVars.keys():
            p[key] = self.tkVars[key].get()

        if self.board_size_disabled.get() > 0:
            del p['BOARD_SIZE']

        self.parent.board.params = p

        # Save transform and area rect
        self.parent.board.param_board_edges = self.parent.imageMask.scaled_mask

        # Detect
        self.parent.detect_stones()

    def log_click_callback(self):
        """Log button click callback"""
        GrLog.show(self.parent.root)

    def debug_click_callback(self):
        """Debug button click callback"""
        if self.debug_dlg is not None:
            self.debug_dlg.close()
            self.debug_dlg = None

        self.debug_dlg = GbrDebugDlg(self.parent)

    def scale_cb_changed(self):
        """Board_size combobox state changed"""
        if self.board_size_disabled is not None and \
            self.board_size_label is not None and \
            self.board_size_scale is not None:

            state = tk.DISABLED if self.board_size_disabled.get() > 0 else tk.NORMAL
            self.board_size_label.config(state = state)
            self.board_size_scale.config(state = state)


    def add_switches(self, rootFrame, params, max_in_row = 6):
        """Add Scale widgets with board parameters"""
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

                # Add a scale from properties
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

                # For board_size, add additional checkbox
                if key == 'BOARD_SIZE':
                    self.board_size_label = label
                    self.board_size_scale = scale

                    state = tk.DISABLED if params[key] is None else tk.NORMAL
                    label.config(state = state)
                    scale.config(state = state)

                    self.board_size_disabled = tk.IntVar()
                    self.board_size_disabled.set(1 if state == tk.DISABLED else 0)
                    cb = tk.Checkbutton(frame,
                                        text = "Automatically detect board size",
                                        variable = self.board_size_disabled,
                                        command = self.scale_cb_changed)
                    #if state == tk.NORMAL: cb.select()
                    cb.grid(row = n, columnspan = 2, padx = 2, pady = 0)
                    n = n + 1

        return vars

    def close(self, update_button_state = True):
        """Graceful way to close the dialog"""
        if self.debug_dlg is not None:
            self.debug_dlg.close()
            self.debug_dlg = None
        if update_button_state: self.parent.buttons['params'].state = False
        self.destroy()

# Stones dialog class
class GbrStonesDlg(GrDialog):
    def __init__(self, parent, *args, **kwargs):
        GrDialog.__init__(self, parent, *args, **kwargs)

    def init_params(self, args, kwargs):
        self.stones = None
        self.get_stones()

    def get_title(self):
        return "Stones"

    def init_frame(self):
        sbr = tk.Scrollbar(self.internalFrame)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        lbox = tk.Listbox(self.internalFrame, yscrollcommand=sbr.set, width = 30)
        if self.stones is not None:
            lbox.insert(tk.END, *self.stones)
        lbox.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        sbr.config(command = lbox.yview)
        lbox.bind('<<ListboxSelect>>', self.select_callback)

    def init_buttons(self):
        self.det_btn_image = ImgButton.get_ui_image("detect_flat.png")
        tk.Button(self.buttonFrame, text = "Detect",
            image = self.det_btn_image, compound="left",
            command = self.detect_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        self.save_btn_image = ImgButton.get_ui_image("save_flat.png")
        tk.Button(self.buttonFrame, text = "Save SGF",
            image = self.save_btn_image, compound="left",
            command = self.save_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        GrDialog.init_buttons(self)
        self.buttonFrame.configure(bd = 0, relief = tk.FLAT)

    def grab_focus(self):
        self.focus_set()

    def update_controls(self):
        pass

    def detect_click_callback(self):
        """Detect button click callback"""
        self.parent.detect_stones()

    def save_click_callback(self):
        """Save button click callback"""
        self.parent.save_sgf()

    def select_callback(self, event):
        """List box selection chang callback"""
        item = event.widget.get(event.widget.curselection()[0])
        parts = item.split(' ')
        bw = 'B' if parts[0].lower() == "black" else 'W'
        a = parts[1][0]
        b = int(parts[1][1:5])
        stone, _ = self.parent.board.find_stone(p = (a,b), f_bw = bw)
        if not stone is None:
            print(stone)

    def close(self, update_button_state = True):
        """Graceful way to close the dialog"""
        if update_button_state: self.parent.buttons['stones'].state = False
        self.destroy()

    def get_stones(self):
        self.stones = []
        bs = self.parent.board.stones
        for bw in bs.keys():
            title = "Black" if bw == "B" else "White"
            for stone in bs[bw]:
                self.stones.extend([title + " " + format_stone_pos(stone)])
        self.stones.sort()


# GUI class
class GbrGUI2(object):

    # Constructor
    def __init__(self, root):
        self.root = root
        self.buttons = dict()
        self.board = GrBoard()
        self.binder = NBinder()
        self.optionsDlg = None
        self.stonesDlg = None

        self.internalFrame = tk.Frame(self.root)
        self.internalFrame.pack(fill = tk.BOTH, expand = True)

        self.__init_menu()
        self.__init_toolbar()
        self.__init_statusbar()
        self.__init_window()

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

        self.buttons['stones'] = ImgButton(toolbarPanel,
            tag = "stones", tooltip = "List of stones", disabled = True,
            callback = self.show_stones_callback)

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
        if self.board.is_gen_board:
            return
        if state:
            self.detect_edges()
            self.imageMask.show()
        else:
            self.imageMask.hide()
        return True

    def show_stones_callback(self, event, tag, state):
        """Show stones button click"""
        if self.board.is_gen_board:
            return
        if self.stonesDlg is not None:
            self.stonesDlg.close(False)
            self.stonesDlg = None
        if state:
            self.stonesDlg = GbrStonesDlg(self)
        return True

    def set_params_callback(self, event, tag, state):
        """Detection params button click"""
        if self.board.is_gen_board:
            return
        if self.optionsDlg is not None:
            self.optionsDlg.close(False)
            self.optionsDlg = None
        if state:
            self.optionsDlg = GbrOptionsDlg(self)
        return True

    def detect_callback(self, event, tag, state):
        """Detect button click"""
        if not self.board.is_gen_board:
            self.detect_stones()
        return False

    def save_sgf_callback(self, event, tag, state):
        """SGF save button click"""
        if not self.board.is_gen_board:
            self.save_sgf()
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
            stone, bw = self.board.find_stone(c = (x, y))
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
        """Load a board image"""

        # Clean up
        GrLog.clear()
        self.imageMask.hide()

        for b in ["edge", "area", "stones", "params", "detect"]:
            self.buttons[b].state = False

        if self.optionsDlg is not None:
            self.optionsDlg.close()
            self.optionsDlg = None
        if self.stonesDlg is not None:
            self.stonesDlg.close()
            self.stonesDlg = None

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


    def detect_edges(self, f_force = False):
        """Detect edges and size of currently loaded board image"""
        if not self.board.param_board_edges is None and not f_force:
           return

        # Process
        GrLog.clear()
        self.board.detect_edges()

        if GrLog.numErrors() > 0:
           self.statusBar.set("Automatic board detection failed, click here for the log")
        else:
            self.statusBar.set("{s}x{s} board detected".format(
                                    s = self.board.board_size))
            self.imageMask.scaled_mask = self.board.param_board_edges
            self.imageMask.size = self.board.board_size

    def detect_stones(self):
        """Detect stones on currently loaded board image"""
        # Process
        GrLog.clear()
        self.board.process()
        self.board.save_params()

        # Update status
        if GrLog.numErrors() > 0:
            self.statusBar.set("Errors during processing, click here for the log")
        else:
            self.statusBar.set("{b} black, {w} white stones on {s}x{s} board detected, click here for the log".format(
                b = len(self.board.black_stones),
                w = len(self.board.white_stones),
                s = self.board.board_size))
            self.buttons['stones'].disabled = False

    def save_sgf(self):
        pass

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
