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
from gr.ui_extra import NLabel, NButton, NBinder, ImgButton, ImagePanel,  \
    StatusPanel, ImageMask, ImageTransform, GrDialog, ImageMarker, \
    treeview_sort_columns
from gr.grlog import GrLog
from gr.utils import format_stone_pos, resize

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
    def __init__(self, *args, **kwargs):
        self.panels = dict()
        GrDialog.__init__(self, *args, **kwargs)

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

    def close(self):
        """Graceful way to close the dialog"""
        if hasattr(self.master, "debug_dlg"):
            self.master.debug_dlg = None
        GrDialog.close(self)

    def save_click_callback(self):
        """Save button click callback"""
        self.save_debug_info()

    def dbg_img_click_callback(self, event):
        """Mouse click on debug info panel callback"""
        w = event.widget
        k = w.tag
        cv2.imshow(k, self.root.board.debug_images[k])

    def on_scroll_configure(self, event):
        """Canvas config callback"""
        self.canvas.configure(scrollregion = self.debugFrame.bbox('all'))

    def add_debug_info(self, parent):
        """Adds debug information images"""
        shape = self.root.board.image.shape
        debug_img = self.root.board.debug_images

        if debug_img is None:
            return

        nrow = 0
        ncol = 0
        sz = min(int(shape[CV_WIDTH] / 2) - 5, 150)

        # Add analysis result images
        for key in sorted(debug_img.keys()):
            frame = tk.Frame(parent)
            frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

            img = resize(debug_img[key], sz)

            imgtk = img_to_imgtk(img)
            panel = NLabel(frame, image = imgtk, tag = key)
            panel.image = imgtk
            panel.grid(row = 0, column = 0)
            panel.bind('<Button-1>', self.dbg_img_click_callback)
            self.panels[key] = panel

            panel = tk.Label(frame, text = key)
            panel.grid(row = 1, column = 0, sticky = "nswe")

            ncol = ncol + 1
            if ncol > 1:
                nrow = nrow + 1
                ncol = 0

    def save_debug_info(self):
        """Saves debug images to given directory"""
        debug_img = self.root.board.debug_images
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

    def update_debug_info(self):
        """Update debug information images"""
        shape = self.root.board.image.shape
        debug_img = self.root.board.debug_images
        sz = min(int(shape[CV_WIDTH] / 2) - 5, 150)

        if debug_img is None:
            return

        for key in debug_img.keys():
            if self.panels.get(key) is not None:
                img = resize(debug_img[key], sz)
                imgtk = img_to_imgtk(img)
                self.panels[key].configure(image = imgtk)
                self.panels[key].image = imgtk


# Options dialog class
class GbrOptionsDlg(GrDialog):
    def __init__(self, *args, **kwargs):
        GrDialog.__init__(self, *args, **kwargs)

    def get_title(self):
        return "Parameters"

    def init_params(self, args, kwargs):
        self.board_size_label = None
        self.board_size_scale = None
        self.board_size_disabled = None
        self.debug_dlg = None

    def init_frame(self):
        self.tkVars = self.add_switches(self.internalFrame, self.root.board.params)

    def init_buttons(self):
        f_top = tk.Frame(self.buttonFrame)
        f_bottom = tk.Frame(self.buttonFrame)
        f_top.pack(side = tk.TOP, fill = tk.BOTH)
        f_bottom.pack(side = tk.BOTTOM, fill = tk.BOTH)

        self.detectVar = tk.IntVar(0)
        tk.Checkbutton(f_top, text = "Detect on parameter changes", variable = self.detectVar,
            command = self.auto_detect_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        NButton(f_bottom,
            text = "Detect", uimage = "detect_flat.png",
            command = self.detect_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        tk.Button(f_bottom, text = "Defaults",
            command = self.default_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        self.log_button = tk.Button(f_bottom, text = "Log",
            state = tk.DISABLED if GrLog.numErrors() == 0 is None else tk.NORMAL,
            command = self.log_click_callback)
        self.log_button.pack(side = tk.LEFT, padx = 5, pady = 5)

        self.dbg_button = tk.Button(f_bottom, text = "Debug",
            state = tk.DISABLED if self.root.board.results is None else tk.NORMAL,
            command = self.debug_click_callback)
        self.dbg_button.pack(side = tk.LEFT, padx = 5, pady = 5)

        tk.Button(f_bottom, text = "Close",
            command = self.close_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        self.bind("<Return>", self.return_callback)

    def update_controls(self):
        if self.dbg_button is not None:
            self.dbg_button.configure(
                state = tk.DISABLED if self.root.board.results is None else tk.NORMAL)
        if self.log_button is not None:
            self.log_button.configure(
                state = tk.DISABLED if GrLog.numErrors() == 0 is None else tk.NORMAL)

    def auto_detect_callback(self):
        """Auto-detect checkbox click callback"""
        if self.detectVar.get() == 0:
            self.root.hide_stones()
        else:
            self.detect_stones(True)
            self.update_controls()

    def detect_click_callback(self):
        """Detect button click callback"""
        self.detect_stones(self.detectVar.get() != 0)
        self.update_controls()

    def default_click_callback(self):
        """Default button click callback"""
        self.root.board.params = dict()

        for key in self.tkVars.keys():
            v = self.root.board.params[key]
            if not v is None: self.tkVars[key].set(v)

        self.board_size_disabled.set(1)
        self.board_size_label.config(state = tk.DISABLED)
        self.board_size_scale.config(state = tk.DISABLED)

        self.root.board.save_params()
        self.update_controls()

    def log_click_callback(self):
        """Log button click callback"""
        GrLog.show(self.root)

    def debug_click_callback(self):
        """Debug button click callback"""
        if self.debug_dlg is not None:
            self.debug_dlg.close()
            self.debug_dlg = None

        self.debug_dlg = GbrDebugDlg(master = self)

    def size_enabled_changed(self):
        """Board_size combobox state changed"""
        if self.board_size_disabled is not None and \
            self.board_size_label is not None and \
            self.board_size_scale is not None:

            state = tk.DISABLED if self.board_size_disabled.get() > 0 else tk.NORMAL
            self.board_size_label.config(state = state)
            self.board_size_scale.config(state = state)

    def scale_changed_callback(self, event):
        """Any scale value changed callback"""
        if self.detectVar.get() > 0:
            self.detect_stones(True)

    def return_callback(self, event):
        self.detect_click_callback()

    def add_switches(self, parent, params, max_in_row = 6):
        """Add Scale widgets with board parameters"""
        n = 1
        ncol = 0
        frame = None
        vars = dict()

        # Add a tabbed notebook
        nb = ttk.Notebook(parent)
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
            keys = sorted(keys, key = lambda k: GR_PARAMS_PROP[k][4] if len(GR_PARAMS_PROP[k]) > 4 else 0)

            for key in keys:
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
                                        variable = v,
                                        command = self.scale_changed_callback)
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
                                        command = self.size_enabled_changed)
                    #if state == tk.NORMAL: cb.select()
                    cb.grid(row = n, columnspan = 2, padx = 2, pady = 0)
                    n = n + 1

        return vars

    def close(self):
        """Graceful way to close the dialog"""
        if self.debug_dlg is not None:
            self.debug_dlg.close()
            self.debug_dlg = None
        if self.detectVar.get() > 0:
            self.root.imageMarker.clear()
        GrDialog.close(self)

    def detect_stones(self, highlight):
        """Detect stones with parameters currenly set"""
        # Get params
        p = dict()
        for key in self.tkVars.keys():
            p[key] = self.tkVars[key].get()

        p['BOARD_EDGES'] = self.root.board.param_board_edges
        p['TRANSFORM'] = self.root.board.param_transform_rect
        if self.board_size_disabled.get() > 0:
            del p['BOARD_SIZE']
            del p['BOARD_EDGES']

        # Detect
        self.root.board.params = p
        self.root.detect_stones(highlight)

        # Update debug info, if the dialog is open
        if self.debug_dlg is not None:
            self.debug_dlg.update_debug_info()

# Stones dialog class
class GbrStonesDlg(GrDialog):
    def __init__(self, *args, **kwargs):
        GrDialog.__init__(self, *args, **kwargs)

    def init_state(self):
        self.stones = None
        self.get_stones()

    def get_minsize(self):
        return (200, 300)

    def get_title(self):
        return "Stones"

    def init_frame(self):
        sbr = tk.Scrollbar(self.internalFrame)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        self.tv = ttk.Treeview(self.internalFrame, column = ("p"),
            height = 10, selectmode = "browse" )
        self.tv.column("#0", width=40, minwidth=20, stretch=tk.NO)
        self.tv.column("p", width=30, minwidth=20, stretch=tk.YES)
        self.tv.heading("#0", text = "Color", anchor = tk.W)
        self.tv.heading("#1", text = "Stone", anchor = tk.W)
        treeview_sort_columns(self.tv)
        self.tv.pack(side = tk.TOP, fill=tk.BOTH, expand = True)
        self.tv.bind("<<TreeviewSelect>>", self.select_callback)

        self.stone_images = [ImgButton.get_ui_image("black_small.png"),
                           ImgButton.get_ui_image("white_small.png")]
        self.update_listbox()

        sbr.config(command = self.tv.yview)
        self.tv.config(yscrollcommand = sbr.set)

    def init_buttons(self):
        NButton(self.buttonFrame, text = "Save SGF",
            uimage = "detect_flat.png", compound="left",
            command = self.save_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        GrDialog.init_buttons(self)
        self.buttonFrame.configure(bd = 0, relief = tk.FLAT)

    def grab_focus(self):
        self.focus_set()

    def update_controls(self):
        """Focus set"""
        self.get_stones()
        self.update_listbox()

    def save_click_callback(self):
        """Save button click callback"""
        self.root.save_sgf_callback(None, "save", True)

    def select_callback(self, event):
        """List box selection chang callback"""
        sel = event.widget.focus()
        if sel is not None and len(sel) > 0:
            item = event.widget.item(sel)
            stone, bw = self.root.board.find_stone(s = item['tags'][0], bw = item['tags'][1])
            if stone is not None:
                self.root.show_stone(stone, bw)

    def close(self):
        """Graceful way to close the dialog"""
        self.root.imageMarker.clear()
        GrDialog.close(self)

    def get_stones(self):
        """Gets a list of stones and formats it for display"""
        bs = self.root.board.stones
        t = [(x, 'W') for x in bs['W']]
        t.extend([(x, 'B') for x in bs['B']])
        ts = sorted(t, key = lambda x: np.sqrt(x[0][GR_A]**2 + x[0][GR_B]**2))
        self.stones = [(format_stone_pos(x[0]), x[1]) for x in ts]

    def update_listbox(self):
        self.tv.delete(*self.tv.get_children())
        for stone in self.stones:
            im = self.stone_images[0] if stone[1] == "B" else self.stone_images[1]
            self.tv.insert("", "end", image = im, values = (stone[0]), tags = (stone))


# GUI class
class GbrGUI2(tk.Tk):

    # Constructor
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, "Go board")
        self.title("Go board")
        self.minsize(300, 400)

        self.buttons = dict()
        self.board = GrBoard()
        self.binder = NBinder()
        self.last_stone = None

        self.internalFrame = tk.Frame(self)
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
            dlg_class = GbrStonesDlg)

        self.buttons['params'] = ImgButton(toolbarPanel,
            tag = "params", tooltip = "Detection params", disabled = True,
            dlg_class = GbrOptionsDlg)

        self.buttons['save'] = ImgButton(toolbarPanel,
            tag = "save", tooltip = "Save as SGF", disabled = True,
            callback = self.save_sgf_callback)

        self.buttons['reset'] = ImgButton(toolbarPanel,
            tag = "reset", tooltip = "Reset image", disabled = True,
            callback = self.reset_image_callback)

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
        self.imageTransform = ImageTransform(self.imagePanel,
            callback = self.end_transform_callback)

        # Image marker(s)
        self.imageMarker = ImageMarker(self.imagePanel, flash = 2)

        ## Mouse move
        ##self.bind('<Motion>', self.mouse_move_callback)


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
        if fn is not None:
            self.load_image(fn)
        return False

    def transform_callback(self, event, tag, state):
        """Transform button click"""
        if not state and self.imageTransform.started:
           self.imageTransform.cancel()
        else:
            self.buttons['area'].state = False
            self.imageMask.hide()
            self.imageMarker.clear()
            self.imageTransform.start()
        return True

    def end_transform_callback(self, t, state):
        """Image transformation complete/cancelled"""
        self.buttons['edge'].state = False
        self.buttons['reset'].disabled = not state
        if state:
            self.board.image = self.imagePanel.image
            self.board.param_transform_rect = t.scaled_rect

    def set_grid_callback(self, event, tag, state):
        """Board grid button click"""
        if self.board.is_gen_board: return
        if state:
            self.imageTransform.cancel()
            self.imageMarker.clear()
            self.detect_edges()
            self.imageMask.show()
        else:
            self.imageMask.hide()
        return True

    def detect_callback(self, event, tag, state):
        """Detect button click"""
        if not self.board.is_gen_board:
            self.detect_stones()
        return False

    def show_stones_callback(self, event, tag, state):
        """Show stones button click"""
        return not self.board.is_gen_board

    def set_params_callback(self, event, tag, state):
        """Detection params button click"""
        return not self.board.is_gen_board

    def save_sgf_callback(self, event, tag, state):
        """SGF save button click"""
        if self.board.results is None:
            return

        fn = filedialog.asksaveasfilename(title = "Save SGF file",
            initialfile = os.path.splitext(self.board.image_file)[0] + '.sgf',
            defaultextension = '.sgf',
            filetypes = (("SGF files","*.sgf"),("All files","*.*")))
        if fn is not None:
            self.save_sgf()

        return False

    def reset_image_callback(self, event, tag, state):
        """Reset button click"""
        self.imageMask.hide()
        self.imageTransform.reset()
        self.board.reset_image()

        self.imagePanel.set_image(self.board.image)
        self.imageMask.scaled_mask = self.board.param_board_edges
        self.imageMask.size = self.board.param_board_size \
            if self.board.param_board_size is not None else DEF_BOARD_SIZE

        self.buttons['reset'].disabled = True
        return False

##    def mouse_move_callback(self, event):
##        x,y = self.winfo_pointerxy()
##        widget = self.winfo_containing(x, y)
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
                self.show_stone(stone, bw)
            else:
                self.statusBar.set("")

    def status_click_callback(self, event):
        """Status bar mouse click"""
        GrLog.show(self)

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
        self.imageMarker.clear()

        for b in ["edge", "area", "stones", "params"]:
            self.buttons[b].release()

        # Load image
        self.board.load_image(filename, f_process = False, f_with_params = True)

        # Display loaded image and mask
        self.imagePanel.set_image(self.board.image)
        self.imageMask.scaled_mask = self.board.param_board_edges
        self.imageMask.size = self.board.param_board_size \
            if self.board.param_board_size is not None else DEF_BOARD_SIZE

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
        if self.board.param_board_edges is not None and \
            self.board.param_board_size is not None and \
            not f_force:
                # Edges/size already detected
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

    def detect_stones(self, highlight = False):
        """Detect stones on currently loaded board image"""
        # Clean up
        GrLog.clear()
        self.imageMarker.clear()

        for b in ["edge", "area"]:
            self.buttons[b].release()

        # Process
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

            self.imageMask.scaled_mask = self.board.board_edges
            self.imageMask.size = self.board.board_size

            for b in ["stones", "save"]:
                self.buttons[b].disabled = False

            if highlight:
                self.show_all_stones()

    def save_sgf(self):
        if not self.board.is_gen_board:
            fn = self.board.save_sgf()
            self.statusBar.set_file("Board saved to ", str(fn))

    def show_stone(self, stone, bw):
        """Highlight one stone"""
        self.imageMarker.clear()
        if not stone is None:
            self.imageMarker.add_stone(stone, bw)
            self.statusBar.set("{} {}".format(
                "Black" if bw == "B" else "White",
                format_stone_pos(stone)))
            self.last_stone = stone

    def show_all_stones(self):
        """Highlight all stones"""
        self.imageMarker.clear()
        self.imageMarker.add_stones(self.board.black_stones, "B")
        self.imageMarker.add_stones(self.board.white_stones, "W")
        self.imageMarker.show()

    def hide_stones(self):
        self.imageMarker.clear()

    #
    # Utility functions
    #


# Main function
def main():
    log = GrLog.init()
    window = GbrGUI2()

    window.mainloop()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
