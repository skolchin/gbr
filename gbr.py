#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Main functions
#
# Author:      kol
#
# Created:     04.07.2019
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

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog  as filedialog
    import ttk
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk

MAX_IMG_SIZE = 500
MAX_DBG_IMG_SIZE = 200

# GUI class
class GbrGUI(object):

    def __init__(self, root, max_img_size = MAX_IMG_SIZE, max_dbg_img_size = MAX_DBG_IMG_SIZE, allow_open = True):
        """Create an instance"""
        self.root, self.max_img_size, self.max_dbg_img_size = root, max_img_size, max_dbg_img_size
        self.imgButtons = dict()
        self.allow_open = allow_open
        self.transform = None

        # Generate board
        self.board = GrBoard()

        # Top-level frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, expand = True, padx = PADX, pady = PADY)
        self.infoFrame = tk.Frame(self.root, width = self.board.image.shape[CV_WIDTH]*2, height = 300)
        self.infoFrame.pack(fill=tk.BOTH, padx = PADX, pady = PADY)
        self.statusFrame = tk.Frame(self.root, width = 200, bd = 1, relief = tk.SUNKEN)
        self.statusFrame.pack(side = tk.BOTTOM, fill=tk.BOTH, padx = PADX, pady = PADY)

        self.__setup_img_frame()
        self.__setup_info_frame()
        self.__setup_status_frame()

    def __setup_img_frame(self):
        """Internal - initialize image frame"""

        # Original image
        self.origImgPanel = addImagePanel(self.imgFrame,
            caption = "Original",
            btn_params = [["area", False, self.set_area_callback],
                          ["reset", False, self.transf_reset_callback, "Reset image"],
                          ['edge', False, self.transform_callback, "Transform image"]],
            image = self.board.image,
            max_size = self.max_img_size,
            frame_callback = self.orig_img_mouse_callback)

        self.imgButtons.update(self.origImgPanel.buttons)
        self.imgButtons['reset'].disabled = True
        self.origImgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Mask on original image
        self.imgMask = ImageMask(self.origImgPanel,
            allow_change = True,
            mask_callback = self.mask_changed_callback)

        # Transform on original image
        self.imgTransform = ImageTransform(self.origImgPanel,
            callback = self.end_transform_callback)

        # Generated image
        self.genImgPanel = addImagePanel(self.imgFrame,
            caption = "Generated",
            btn_params = [["box", False, self.show_stones_callback, "Show/hide detection boxes"],
                          ["white", True, self.show_stones_callback, "Show/hide white stones"],
                          ["black", True, self.show_stones_callback, "Show/hide black stones"]],
            image = self.board.image,
            max_size = self.max_img_size,
            frame_callback = self.gen_img_mouse_callback)
        self.imgButtons.update(self.genImgPanel.buttons)
        self.genImgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Debug images
        self.dbgPanel = addImagePanel(self.imgFrame,
            caption = "Analysis",
            frame_callback = self.dbg_img_mouse_callback,
            scrollbars = (False, True))
        self.dbgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.dbgFrame = tk.Frame(self.dbgPanel.canvas)
        self.dbgPanel.canvas.create_window((0,0), window=self.dbgFrame, anchor='nw')
        self.dbgFrame.bind('<Configure>', self.on_scroll_configure)


    def __setup_info_frame(self):
        """Internal - initialize button and settings frame"""
        self.buttonFrame = tk.Frame(self.infoFrame, bd = 1, relief = tk.RAISED,
                                               width = self.max_img_size*2, height = 50)
        self.buttonFrame.grid(row = 0, column = 0, sticky = "nswe")
        self.buttonFrame.pack_propagate(0)

        if self.allow_open:
            self.loadImgBtn = tk.Button(self.buttonFrame, text = "Load image",
                                                          command = self.load_img_callback)
            self.loadImgBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.saveParamBtn = tk.Button(self.buttonFrame, text = "Save params",
                                                        command = self.save_json_callback)
        self.saveParamBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.saveBrdBtn = tk.Button(self.buttonFrame, text = "Save board",
                                                      command = self.save_jgf_callback)
        self.saveBrdBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.applyBtn = tk.Button(self.buttonFrame, text = "Detect",
                                                    command = self.apply_callback)
        self.applyBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.applyDefBtn = tk.Button(self.buttonFrame, text = "Defaults",
                                                       command = self.apply_def_callback)
        self.applyDefBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        self.showLogBtn = tk.Button(self.buttonFrame, text = "Show log",
                                                       command = self.show_log_callback)
        self.showLogBtn.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        # Info frame: stones info
        self.boardInfo = tk.StringVar()
        self.boardInfo.set("No stones found")
        self.boardInfoPanel = tk.Label(self.buttonFrame, textvariable = self.boardInfo)
        self.boardInfoPanel.pack(side = tk.LEFT, padx = PADX)

        # Info frame: switches
        self.switchFrame = tk.Frame(self.infoFrame, bd = 1, relief = tk.RAISED)
        self.switchFrame.grid(row = 1, column = 0, sticky = "nswe")
        self.tkVars = self.add_switches(self.switchFrame, self.board.params)

    def __setup_status_frame(self):
        """Internal - initialize status bar"""
        self.statusInfo = addStatusPanel(self.statusFrame, self.max_img_size*2)
        self.statusInfo.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)


    def gen_img_mouse_callback(self, event):
        """Handler for mouse click on generated image panel"""
        if self.board.is_gen_board:
            return

        x, y = self.genImgPanel.frame2image((event.x, event.y))
        p, f = self.board.find_stone(coord = (x,y))
        if not p is None:
            fs = "Black"
            if f == GR_STONES_W: fs = "White"
            ct = "{f} {a}{b} at ({x},{y}):{r}".format(
                f = fs,
                a = format_stone_pos(p, GR_A),
                b = format_stone_pos(p, GR_B),
                x = round(p[GR_X],0),
                y = round(p[GR_Y],0),
                r = round(p[GR_R],0))
            print(ct)
            self.statusInfo.set(ct)

    def orig_img_mouse_callback(self, event):
        """Handler for mouse click on original image panel"""
        if not self.imgButtons['edge'].state and not self.imgButtons['area'].state:
           self.load_img_callback()

    def dbg_img_mouse_callback(self, event):
        """Handler for mouse click on debug info panel"""
        if self.board.is_gen_board:
            return

        w = event.widget
        k = w.tag
        cv2.imshow(k, self.board.debug_images[k])

    def load_img_callback(self):
        """Load image button/click handler"""
        if not self.allow_open: return  # GUI used from other app

        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "": self.load_image(fn)

    def save_json_callback(self):
        """Save params button handler"""
        if self.board.is_gen_board:
            # Generated board - nothing to do!
            return
        else:
            fn = self.board.save_params()
            self.statusInfo.set_file("Params saved to ", str(fn))

    def save_jgf_callback(self):
        """Save JGF button handler"""
        if self.board.is_gen_board:
            # Generated board - nothing to do!
            return

        fn = self.board.save_board_info()
        self.statusInfo.set_file("Board saved to ", str(fn))

    def apply_callback(self):
        """Apply param changes button handler"""
        if self.board.is_gen_board:
            return

        p = dict()
        for key in self.tkVars.keys():
            p[key] = self.tkVars[key].get()
        self.board.params = p
        self.update_board(reprocess = True)
        if GrLog.numErrors() == 0:
            self.statusInfo.set("No errors")

    def apply_def_callback(self):
        """Apply defaults button handler"""
        if self.board.is_gen_board:
            return

        p = DEF_GR_PARAMS.copy()
        self.board.params = p
        for key in self.tkVars.keys():
            self.tkVars[key].set(p[key])
        self.update_board(reprocess = True)
        if GrLog.numErrors() == 0:
            self.statusInfo.set("No errors")

    def show_log_callback(self):
        """Show log button handler"""
        GrLog.show(self.root)

    def on_scroll_configure(self, event):
        """Canvas config callback"""
        self.dbgPanel.canvas.configure(scrollregion = self.dbgFrame.bbox('all'))

    def show_stones_callback(self, event, tag, state):
        """Stone visibility buttons handler"""
        if self.board.is_gen_board:
            return False
        else:
            self.update_board(reprocess = False)
            return True

    def set_area_callback(self, event, tag, state):
        """Set board area button handler"""
        if self.board.is_gen_board:
            return False
        else:
            if state:
               self.imgMask.show()
            else:
               self.imgMask.hide()
            return True

    def mask_changed_callback(self, mask):
        """Callback for ImageMask end dragging event"""
        self.board.area_mask = mask.scaled_mask

    def transform_callback(self, event, tag, state):
        """Transform button handler"""
        if self.board.is_gen_board:
           return False

        if not state and self.imgTransform.started:
           self.imgTransform.cancel()
        else:
            self.origImgPanel.buttons['edge'].state = False
            self.imgMask.hide()
            self.imgTransform.start()
        return True

    def end_transform_callback(self, t, state):
        """Callback for transformation ending"""
        self.origImgPanel.buttons['edge'].state = False
        if state:
           self.origImgPanel.buttons['reset'].disabled = not state

           self.board.image = t.image
           self.board.transform_rect = t.scaled_rect
           self.update_board(True)

    def transf_reset_callback(self, event, tag, state):
        """Transformed image reset button handler"""
        self.imgMask.hide()
        self.imgTransform.cancel()

        #self.imgTransform.reset()
        self.origImgPanel.image = self.board.src_image
        self.update_board(True)

        self.origImgPanel.buttons['edge'].state = False
        self.origImgPanel.buttons['reset'].disabled = True
        return False

    # Add Scale widgets with board recognition parameters
    def add_switches(self, rootFrame, params, nrow = 0):
        n = 1
        ncol = 0
        frame = None
        vars = dict()

        # Add a tabbed notebook
        nb = ttk.Notebook(rootFrame)
        nb.grid(row = nrow, column = 0, sticky = "nswe", padx = PADX, pady = PADY)

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
            for key in sorted(keys):
                if (n == 3 or frame is None):
                    frame = tk.Frame(nbFrame, width = 400)
                    frame.grid(row = 0, column = ncol, padx = 3, pady = 3)
                    n = 0
                    ncol = ncol + 1

                # Add a switch
                panel = tk.Label(frame, text = key)
                panel.grid(row = n, column = 0, padx = 2, pady = 2, sticky = "s")

                v = tk.IntVar()
                v.set(params[key])
                panel = tk.Scale(frame, from_ = GR_PARAMS_PROP[key][0],
                                        to = GR_PARAMS_PROP[key][1],
                                        orient = tk.HORIZONTAL,
                                        variable = v)
                panel.grid(row = n, column = 1, padx = 2, pady = 2)
                vars[key] = v

                n = n + 1
        return vars

    # Add analysis results info
    def add_debug_info(self, root, shape, debug_img, debug_info):
        if debug_img is None:
            return

        nrow = 0
        ncol = 0
        sx = min(int(shape[CV_WIDTH] / 2) - 5, self.max_dbg_img_size)
        sy = int(float(sx) / float(shape[CV_WIDTH]) * shape[CV_HEIGTH])

        # Remove all previously added controls
        for c in root.winfo_children():
            c.destroy()

        # Add analysis result images
        for key in sorted(debug_img.keys()):
            frame = tk.Frame(root)
            frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

            img = cv2.resize(debug_img[key], (sx, sy))

            imgtk = img_to_imgtk(img)
            panel = NLabel(frame, image = imgtk, tag = key)
            panel.image = imgtk
            panel.grid(row = 0, column = 0)
            panel.bind('<Button-1>', self.dbg_img_mouse_callback)

            panel = tk.Label(frame, text = key)
            panel.grid(row = 1, column = 0, sticky = "nswe")

            ncol = ncol + 1
            if ncol > 1:
                nrow = nrow + 1
                ncol = 0

        # Add text information
        frame = tk.Frame(root)
        frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

        lbox = tk.Listbox(frame)
        lbox.grid(row = 0, column = 0, sticky = "nswe")
        lbox.config(width = int(sx / 8))

        for key in sorted(debug_info.keys()):
            lbox.insert(tk.END, "{}: {}".format(key, debug_info[key]))

        panel = tk.Label(frame, text = "TEXT_INFO")
        panel.grid(row = 1, column = 0, sticky = "nswe")

    # Update board
    def update_board(self, reprocess = True):
        # Process original image
        if self.board.results is None or reprocess:
            GrLog.clear()
            try:
                self.board.process()
                if GrLog.numErrors() > 0:
                    self.statusInfo.set("Errors during processing, see the log")
            except:
                logging.exception("Error")
                self.statusInfo.set("Error during processing, see the log")
                return

        # Generate board using analysis results
        btn_state = dict()
        for key in self.imgButtons.keys():
            btn_state[key] = self.imgButtons[key].state
        self.genImgPanel.image = self.board.show_board(show_state = btn_state)

        if self.board.results is None:
            self.boardInfo.set("")
        else:
            board_size = self.board.board_size
            black_stones = self.board.black_stones
            white_stones = self.board.white_stones

            self.boardInfo.set("Board size: {}, black stones: {}, white stones: {}".format(
                                      board_size, black_stones.shape[0], white_stones.shape[0]))

        # Update params
        p = self.board.params
        for key in p.keys():
            if key in self.tkVars.keys():
                self.tkVars[key].set(p[key])

        # Update debug info
        self.add_debug_info(self.dbgFrame, self.board.board_shape,
                                           self.board.debug_images, self.board.debug_info)

    # Load specified image
    def load_image(self, fn):
        # Load the image
        GrLog.clear()
        try:
            params_loaded = self.board.load_image(fn, f_with_params = True)
            self.origImgPanel.image = self.board.image
            self.imgMask.scaled_mask = self.board.area_mask
            self.imgButtons['reset'].disabled = not self.board.can_reset_image

            # Reset button state to default
            for key in self.imgButtons.keys():
                self.imgButtons[key].state = False
            self.imgButtons['black'].state = True
            self.imgButtons['white'].state = True

            # Process image
            self.update_board(reprocess = False)

            # Update status
            if GrLog.numErrors() > 0:
                self.statusInfo.set("Errors during file loading, see the log")
            else:
                ftitle = ": "
                if params_loaded: ftitle = " (with params): "
                self.statusInfo.set_file("File loaded" + ftitle, self.board.image_file)

        except:
            logging.exception("Error")
            self.statusInfo.set("Error when loading image, see the log")


# Main function
def main():
    # Construct interface
    window = tk.Tk()
    window.title("Go board")

    log = GrLog.init()
    gui = GbrGUI(window)

    #window.grid_columnconfigure(0, weight=1)
    #window.grid_rowconfigure(0, weight=1)
    #window.resizable(True, True)

    # Main loop
    window.mainloop()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

