#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Main functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------
from gr.board import GrBoard
from gr.utils import img_to_imgtk, resize2, format_stone_pos
from gr.grdef import *
from gr.ui_extra import *

import numpy as np
import cv2
import sys
import os
from PIL import Image, ImageTk

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog  as filedialog
    import ttk
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk

MAX_IMG_SIZE = 550
MAX_DBG_IMG_SIZE = 200
DEF_BTN_STATE = { "black": True, "white": True, "box": False, "edge": False }

# GUI class
class GbrGUI(object):

    def __init__(self, root, max_img_size = MAX_IMG_SIZE, max_dbg_img_size = MAX_DBG_IMG_SIZE, allow_open = True):
        # Params
        self.root, self.max_img_size, self.max_dbg_img_size = root, max_img_size, max_dbg_img_size
        self.buttonState = DEF_BTN_STATE.copy()
        self.imgButtons = dict()
        self.allow_open = allow_open

        # Generate board
        self.board = GrBoard()

        # Prepare images
        self.origImgTk, self.zoom = self.make_imgtk(self.board.image)
        self.genImgTk = self.origImgTk

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
        _, self.origImgPanel, buttons = \
                addImagePanel(self.imgFrame,
                          "Original",
                          [["edge", False, self.set_edges_callback]],
                          self.origImgTk,
                          self.orig_img_mouse_callback)
        self.imgButtons.update(buttons)

        _, self.genImgPanel, buttons = \
                addImagePanel(self.imgFrame,
                          "Generated",
                           [["box", False, self.show_stones_callback, "Show/hide detection boxes"],
                           ["white", True, self.show_stones_callback, "Show/hide white stones"],
                           ["black", True, self.show_stones_callback, "Show/hide black stones"]],
                          self.genImgTk,
                          self.gen_img_mouse_callback)
        self.imgButtons.update(buttons)

        self.dbgPanel, self.dbgCanvasFrame, buttons = \
                addImagePanel(self.imgFrame,
                          "Analysis",
                           [],
                           None,
                           self.dbg_img_mouse_callback)


        # Add canvas
        self.dbgCanvas = tk.Canvas(self.dbgCanvasFrame, width = self.max_dbg_img_size*2+10,
                                                        height = self.origImgTk.height())
        scroll = tk.Scrollbar(self.dbgCanvasFrame, command = self.dbgCanvas.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady = PADY)
        self.dbgCanvas.configure(yscrollcommand = scroll.set)
        self.dbgCanvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.dbgFrame = tk.Frame(self.dbgCanvas)
        self.dbgCanvas.create_window((0,0), window=self.dbgFrame, anchor='nw')
        self.dbgFrame.bind('<Configure>', self.on_scroll_configure)


    def __setup_info_frame(self):
        # Info frame: buttons
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
        # Status bar
        self.stoneInfo = tk.StringVar()
        self.stoneInfo.set("")
        self.stoneInfoPanel = tk.Label(self.statusFrame, textvariable = self.stoneInfo)
        self.stoneInfoPanel.grid(row = 0, column = 0, sticky = tk.W, padx = 5, pady = 2)


    # Callback functions
    # Callback for mouse events on generated board image
    def gen_img_mouse_callback(self, event):
        # Convert from widget coordinates to image coordinates
        w = event.widget.winfo_width()
        h = event.widget.winfo_height()
        x = event.x
        y = event.y
        zx = self.zoom[GR_X]
        zy = self.zoom[GR_Y]

        if self.board.is_gen_board:
            return

        x = int((x - (w - self.board.board_shape[CV_WIDTH] * zx) / 2) / zx)
        y = int((y - (h - self.board.board_shape[CV_HEIGTH] * zy) / 2) / zy)
        print('{}, {}'.format(x, y))

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
           self.stoneInfo.set(ct)

    # Callback for mouse events on original image
    def orig_img_mouse_callback(self, event):
        self.load_img_callback()

    # Callback for mouse event on debug image
    def dbg_img_mouse_callback(self, event):
        if self.board.is_gen_board:
           return

        w = event.widget
        k = w.tag
        cv2.imshow(k, self.board.debug_images[k])

    # Load image button callback
    def load_img_callback(self):
        if not self.allow_open: return  # GUI used from other app

        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "": self.load_image(fn)

    # Save params button callback
    def save_json_callback(self):
        # Save json with current parsing parameters
        if self.board.is_gen_board:
            # Generated board - nothing to do!
            return
        else:
            fn = self.board.save_params()
            self.stoneInfo.set("Params saved to: " + str(fn))

    # Save stones button callback
    def save_jgf_callback(self):
        # Save json with current parsing parameters
        if self.board.is_gen_board:
            # Generated board - nothing to do!
            return

        fn = self.board.save_board_info()
        self.stoneInfo.set("Board saved to: " + str(fn))

    # Apply button callback
    def apply_callback(self):
        p = dict()
        for key in self.tkVars.keys():
            p[key] = self.tkVars[key].get()
        self.board.params = p
        self.update_board(reprocess = True)

    # Apply defaults button callback
    def apply_def_callback(self):
        p = DEF_GR_PARAMS.copy()
        self.board.params = p
        for key in self.tkVars.keys():
            self.tkVars[key].set(p[key])
        self.update_board(reprocess = True)

    # Callback for canvas configuration
    def on_scroll_configure(self, event):
        self.dbgCanvas.configure(scrollregion = self.dbgFrame.bbox('all'))

    # Callback for "Show stones/edges"
    def show_stones_callback(self, event, tag, state):
        if self.board.is_gen_board:
            return False
        else:
            self.buttonState[tag] = state
            self.update_board(reprocess= False)
            return True

    # Callback for "Set edges"
    def set_edges_callback(self, event, tag, state):
        if self.board.is_gen_board:
            return False
        else:
            self.buttonState[tag] = state
            return True

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
            for key in params.keys():
                if GR_PARAMS_PROP[key][2] == tab:
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
        sx = int(shape[CV_WIDTH] / 2) - 5
        if sx > self.max_dbg_img_size: sx = self.max_dbg_img_size
        sy = int(float(sx) / float(shape[CV_WIDTH]) * shape[CV_HEIGTH])

        # Remove all previously added controls
        for c in root.winfo_children():
            c.destroy()

        # Add analysis result images
        for key in sorted(debug_img.keys()):
            frame = tk.Frame(root)
            frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

            img = cv2.resize(debug_img[key], (sx, sy))

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
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
            self.board.process()

        # Generate board using analysis results
        self.genImg = self.board.show_board(show_state = self.buttonState)
        self.genImgTk, _ = self.make_imgtk(self.genImg)
        self.genImgPanel.configure(image = self.genImgTk)

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

    # Convert origImg to ImageTk
    # If image size greater than maximim one, resize it to proper level and store zoom factor
    def make_imgtk(self, img):
        imgtk = None
        z = [1.0,1.0]
        if img.shape[0] <= self.max_img_size and img.shape[1] <= self.max_img_size:
           imgtk = img_to_imgtk(img)
        else:
           img2, z = resize2(img, self.max_img_size, False)
           imgtk = img_to_imgtk(img2)
        return imgtk, z

    # Load specified image
    def load_image(self, fn):
        # Load the image
        params_loaded = self.board.load_image(fn, f_with_params = True)
        self.origImgTk, self.zoom = self.make_imgtk(self.board.image)
        self.origImgPanel.configure(image = self.origImgTk)

        # Reset button state to default
        self.buttonState = DEF_BTN_STATE.copy()
        for key in self.buttonState:
            self.imgButtons[key].state = self.buttonState[key]

        # Process image
        self.update_board(reprocess = False)

        # Update status
        ftitle = ""
        if params_loaded: ftitle = " (with params)"
        self.stoneInfo.set("File loaded{ft}: {fn}".format(ft = ftitle, fn = self.board.image_file))

# Main function
def main():
    # Construct interface
    window = tk.Tk()
    window.title("Go board")

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

