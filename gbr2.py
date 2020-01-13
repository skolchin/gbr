
#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     New GBR UI
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019-2020
# Licence:     MIT
#-------------------------------------------------------------------------------

from gr.board import GrBoard
from gr.grdef import *
from gr.ui_extra import *
from gr.binder import NBinder
from gr.log import GrLogger
from gr.utils import format_stone_pos, resize, img_to_imgtk, dict_value2key
from gr.grq import BoardOptimizer
from gr.gr import convert_xy

import numpy as np
import cv2
import os
from PIL import Image, ImageTk
from threading import Thread, Lock
from collections import namedtuple

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

    def init_frame(self, internalFrame):
        sbr = tk.Scrollbar(internalFrame)
        self.canvas = tk.Canvas(internalFrame, yscrollcommand=sbr.set)
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        sbr.pack(side = tk.RIGHT, fill = tk.Y)
        sbr.config(command = self.canvas.yview)

        self.debugFrame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window = self.debugFrame, anchor='nw')
        self.add_debug_info(self.debugFrame)

        self.canvas.bind('<Configure>', self.on_scroll_configure)

    def init_buttons(self, buttonFrame):
        tk.Button(buttonFrame, text = "Save images",
                  command = self.save_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)
        GrDialog.init_buttons(self, buttonFrame)

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
        self.last_params = None

        self.max_iter = 100

        self.lock = Lock()
        self.qc_thread = None
        self.qc = None
        self.qc_log = None
        self.optimize_cancel = False

    def init_frame(self, internalFrame):
        self.tkVars = self.add_switches(internalFrame)
        self.add_optimization()

    def init_buttons(self, buttonFrame):
        f_top = tk.Frame(buttonFrame)
        f_bottom = tk.Frame(buttonFrame)
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

        self.resetButton = NButton(f_bottom,
            text = "Revert", uimage = "reset_flat.png", state = tk.DISABLED,
            command = self.reset_click_callback)
        self.resetButton.pack(side = tk.LEFT, padx = 5, pady = 5)

        self.log_button = tk.Button(f_bottom, text = "Log",
            state = tk.DISABLED if self.root.log.errors == 0 is None else tk.NORMAL,
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
        if self.qc_thread is not None:
            # Optimization thread is running, don't update state
            return
        if self.dbg_button is not None:
            self.dbg_button.configure(
                state = tk.DISABLED if self.root.board.results is None else tk.NORMAL)
        if self.log_button is not None:
            self.log_button.configure(
                state = tk.DISABLED if self.root.log.errors == 0 is None else tk.NORMAL)

    def close(self):
        """Graceful way to close the dialog"""
        if self.debug_dlg is not None:
            self.debug_dlg.close()
            self.debug_dlg = None
        if self.detectVar.get() > 0:
            self.root.imageMarker.clear()
        if not self.qc_thread is None:
            self.optimize_cancel = True
        GrDialog.close(self)

    def auto_detect_callback(self):
        """Auto-detect checkbox click callback"""
        if self.detectVar.get() == 0:
            self.root.hide_stones()
        else:
            self.detect_stones(highlight = True)
            self.update_controls()

    def detect_click_callback(self):
        """Detect button click callback"""
        self.detect_stones(highlight = self.detectVar.get() != 0)
        self.update_controls()

    def default_click_callback(self):
        """Default button click callback"""
        self.root.board.params.reset()
        self.root.board.stones.clear(with_forced = True)
        self.update_switches()

        self.board_size_disabled.set(1)
        self.board_size_label.config(state = tk.DISABLED)
        self.board_size_scale.config(state = tk.DISABLED)

        self.root.board.save_params()
        self.update_controls()

        self.root.boardArea.mask = None
        self.root.boardGrid.mask = None

    def log_click_callback(self):
        """Log button click callback"""
        if self.nb.index(self.nb.select()) == 3 and self.qc_log is not None:
            self.qc_log.show()
        else:
            self.root.log.show()

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

    def optimize_click_callback(self):
        """Optimize button click"""

        # Use lock to prevent simultaneous access to the same variables
        with self.lock:
            if not self.qc_thread is None and self.qc_thread.is_alive():
                # Seems that optimization is already running, try to cancel
                self.progressLabel.set("Cancelling")
                self.optimize_cancel = True
            else:
                # Initialize optimizer
                self.qc = BoardOptimizer(board = GrBoard(), debug = False)
                self.qc.board.image = self.root.board.image
                self.qc.board.params = self.root.board.params
                self.last_params = self.root.board.params.todict()
                self.resetButton.configure(state = tk.NORMAL)
                self.qc_log = None

                # Launch a separate thread and return
                self.qc_thread = Thread(target = self.optimizer_function)
                self.qc_thread.start()

    def optimizer_function(self):
        """Optimization function (runs in separate thread)"""
        if self.qc is None:
            return

        # Prepare
        with self.lock:
            self.set_controls_state(tk.DISABLED)
            self.optimize_cancel = False
            self.optimizeButton.configure(text = "Cancel")
            self.progressLabel.set("Running detection")
            self.progressVar.set(0)

        # Run
        success = self.qc.optimize(groups = [1, 2],
            max_pass = self.max_iter,
            callback = self.optimize_callback)

        # Clean up
        with self.lock:
            if not self.optimize_cancel and success:
                self.root.board.params = self.qc.board.params
                self.update_switches()

            self.optimizeButton.configure(text = "Auto-detect")
            self.progressLabel.set("Completed {}".format("successfully"
                if success else "unsuccessfully"))
            self.set_controls_state(tk.ACTIVE)
            self.qc_log = self.qc.log
            self.qc_thread = None
            self.qc = None

    def optimize_callback(self, params):
        """Optimize callback"""
        with self.lock:
            if self.optimize_cancel:
                self.progressVar.set(0)
                return True
            else:
                npass = params['npass']
                self.progressLabel.set("Running detection: pass {} of {}".format(npass, self.max_iter))
                self.progressVar.set(npass)
            return False

    def reset_click_callback(self):
        """Reset params button click"""
        if self.last_params is not None:
            self.root.board.params = self.last_params
            self.update_switches()

            self.last_params = None
            self.resetButton.configure(state = tk.DISABLED)

            if self.detectVar.get() > 0:
                self.detect_stones(highlight = True)
                self.update_controls()


    def add_switches(self, parent, max_in_row = 6):
        """Add Scale widgets for changing board parameters"""
        n = 1
        ncol = 0
        frame = None
        vars = dict()

        # Add a tabbed notebook
        self.nb = ttk.Notebook(parent)
        self.nb.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        # Add scales for board parameters to notebook tabs
        for tab in self.root.board.params.groups:
            # Add a tab frame
            nbFrame = tk.Frame(self.nb, width = 400)
            self.nb.add(nbFrame, text = tab)
            frame = None
            n = 0
            ncol = 0

            # Iterate through the params processing only ones belonging to current tab
            for param in self.root.board.params.group_params(tab):
                if (n == max_in_row or frame is None):
                    frame = tk.Frame(nbFrame, width = 400)
                    frame.grid(row = 0, column = ncol, padx = 3, pady = 3)
                    n = 0
                    ncol = ncol + 1

                # Add a scale from properties
                label = tk.Label(frame, text = param.title)
                label.grid(row = n, column = 0, padx = 2, pady = 0, sticky = "s", ipady=4)

                v = tk.IntVar()
                v.set(param.v)
                scale = tk.Scale(frame, from_ = param.min_v,
                                        to = param.max_v,
                                        orient = tk.HORIZONTAL,
                                        variable = v,
                                        command = self.scale_changed_callback)
                scale.grid(row = n, column = 1, padx = 2, pady = 0)
                vars[param.key] = v
                n = n + 1

                # For board_size, add additional checkbox
                if param.key == 'BOARD_SIZE':
                    self.board_size_label = label
                    self.board_size_scale = scale

                    state = tk.DISABLED if param.v is None else tk.NORMAL
                    label.config(state = state)
                    scale.config(state = state)

                    self.board_size_disabled = tk.IntVar()
                    self.board_size_disabled.set(1 if state == tk.DISABLED else 0)
                    cb = tk.Checkbutton(frame,
                                        text = "Automatically detect board size",
                                        variable = self.board_size_disabled,
                                        command = self.size_enabled_changed)
                    cb.grid(row = n, columnspan = 2, padx = 2, pady = 0)
                    n = n + 1

        return vars

    def update_switches(self):
        """Update controls by changed board parameter values"""
        for k in self.root.board.params:
            if k in self.tkVars:
                self.tkVars[k].set(self.root.board.params[k])

    def add_optimization(self):
        # Add "optimization" tab to notebook
        nbFrame = tk.Frame(self.nb, width = 400)
        self.nb.add(nbFrame, text = "Optimization")

        frame = tk.Frame(nbFrame, bd = 20, relief = tk.FLAT)
        frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        label = tk.Label(frame,
            text = "Press the button below to initiate automatic selection of\n" +
                    "stone recognition parameters\n\nMake sure that " +
                    "board area and grid are defined and\n" +
                    "proper board size set or detected" )
        label.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        frame = tk.Frame(nbFrame, bd = 5)
        frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        self.optimizeButton = tk.Button(frame, text = "Auto-detect",
            takefocus = False,
            command = self.optimize_click_callback)
        self.optimizeButton.pack(anchor = tk.CENTER)

        frame = tk.Frame(nbFrame, bd = 5)
        frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        self.progressLabel = tk.StringVar("")
        label = tk.Label(frame, textvariable = self.progressLabel)
        label.pack(side = tk.TOP, fill = tk.BOTH, expand = True, pady = 5)

        self.progressVar = tk.IntVar(0)
        progress = ttk.Progressbar(frame,
            orient = "horizontal", maximum = self.max_iter,
            mode = "determinate", length = 300,
            variable = self.progressVar)
        progress.pack(anchor = tk.CENTER, pady = 5)

    def detect_stones(self, highlight):
        """Detect stones with parameters currenly set"""

        # Save parameters for resetting
        if self.last_params is None:
            self.last_params = self.root.board.params.todict()
            self.resetButton.configure(state = tk.NORMAL)

        # Get params
        p = dict()
        for key in self.tkVars.keys():
            p[key] = self.tkVars[key].get()
        self.root.board.params = p

        if self.board_size_disabled.get() > 0:
            self.root.board.param_board_size = None
            self.root.board.param_board_edges = None

        # Detect
        self.root.detect_stones(highlight)

        # Update debug info, if the dialog is open
        if self.debug_dlg is not None:
            self.debug_dlg.update_debug_info()

    def set_controls_state(self, state):
        """Enables/disables all controls"""
        for w0 in self.buttonFrame.winfo_children():
            for w2 in w0.winfo_children():
                if isinstance(w2, tk.Button) or isinstance(w2, tk.Checkbutton):
                    w2.configure(state = state)

        for w in self.root.toolbarPanel.winfo_children():
            if isinstance(w, ImgButton):
                w.configure(state = state)

# Stone properties dialog
class GbrChangeStoneDlg(GrDialog):
    def init_params(self, args, kwargs):
        self.stone = None
        self.vars = None
        self.frame = None
        self.is_new = False
        self.ok_cancel = True

    def get_minsize(self):
        return (200, 200)

    def get_title(self):
        return "Change stone"

    def get_position(self):
        if self.stone is None:
            return GrDialog.get_position(self)
        else:
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            p = self.root.imagePanel.image2frame((self.stone[GR_X], self.stone[GR_Y]))
            return (x + p[0] + 40, y + p[1] + 40)

    def get_offset(self):
        """Override to define offset from to parent window"""
        return (15, 350)

    def init_frame(self, internalFrame):
        self.set_vars()
        self.set_controls(internalFrame)

    def ok_click_callback(self):
        self.stone[GR_X] = self.vars['x'].get()
        self.stone[GR_Y] = self.vars['y'].get()
        self.stone[GR_R] = self.vars['r'].get()
        self.stone[GR_BW] = dict_value2key(STONE_COLORS, self.vars['bw'].get())
        GrDialog.ok_click_callback(self)

    def select_pos_callback(self, event):
        event.cancel = True

    def close(self):
        GrDialog.close(self)

    def set_vars(self):
        if self.vars is None:
            self.vars = {
                'pos': tk.StringVar(),
                'x': tk.IntVar(),
                'y': tk.IntVar(),
                'r': tk.IntVar(),
                'bw': tk.StringVar()
                }
        if self.stone is not None:
            self.vars['pos'].set(format_stone_pos(self.stone))
            self.vars['x'].set(self.stone[GR_X])
            self.vars['y'].set(self.stone[GR_Y])
            self.vars['r'].set(self.stone[GR_R])
            self.vars['bw'].set(STONE_COLORS[self.stone[GR_BW]])

    def set_controls(self, internalFrame):
        if self.frame is None:
            # Add controls 1st time
            self.frame = tk.Frame(internalFrame, bd = 8)
            self.frame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

            # Stone position
            tk.Label(self.frame,
                text = "Stone").grid(row = 0, column = 0,
                    padx = 2, pady = 0, sticky = "s", ipady=4)
            pos_frame = tk.Frame(self.frame)
            pos_frame.grid(row = 0, column = 1,
                    padx = 2, pady = 0, sticky = "s", ipady=4)

            tk.Label(pos_frame,
                anchor = tk.W,
                justify = tk.LEFT,
                textvariable = self.vars['pos']).pack(side = tk.LEFT,
                fill = tk.BOTH, expand = True)
            self.posBtn = ImgButton(pos_frame,
                tag = "cross_small", tooltip = "Select position",
                disabled = not self.is_new, command = self.select_pos_callback)
            self.posBtn.pack(side = tk.LEFT, padx = 3)

            # X
            tk.Label(self.frame,
                text = "x").grid(row = 1, column = 0,
                    padx = 2, pady = 0, sticky = "s", ipady=4)
            tk.Scale(self.frame,
                from_ = 0,
                to = self.root.board.image.shape[CV_WIDTH]-1,
                orient = tk.HORIZONTAL,
                width = 8,
                variable = self.vars['x']).grid(row = 1, column = 1,
                padx = 2, pady = 0, sticky = "s", ipady=4)

            # Y
            tk.Label(self.frame,
                text = "y").grid(row = 1, column = 3,
                    padx = 2, pady = 0, sticky = "s", ipady=4)
            tk.Scale(self.frame,
                from_ = 0,
                to = self.root.board.image.shape[CV_WIDTH]-1,
                orient = tk.HORIZONTAL,
                width = 8,
                variable = self.vars['y']).grid(row = 1, column = 4,
                padx = 2, pady = 0, sticky = "s", ipady=4)

            # R
            tk.Label(self.frame,
                text = "Radius").grid(row = 2, column = 0,
                    padx = 2, pady = 0, sticky = "s", ipady=4)
            tk.Scale(self.frame,
                from_ = 5,
                to = 40,
                orient = tk.HORIZONTAL,
                width = 8,
                variable = self.vars['r']).grid(row = 2, column = 1,
                padx = 2, pady = 0, sticky = "s", ipady=4)

            # BW
            tk.Label(self.frame,
                text = "Color").grid(row = 2, column = 3,
                    padx = 2, pady = 0, sticky = "s", ipady=4)

            ttk.Combobox(self.frame,
                values = ["Black", "White"],
                width = 10,
                textvariable = self.vars['bw']).grid(row = 2, column = 4,
                padx = 2, pady = 0, sticky = "s", ipady=4)

        self.enable_controls(not self.stone is None, "readonly")
        self.posBtn.disabled = not self.is_new


    def set_stone(self, stone):
        self.stone = stone.copy()
        self.set_vars()
        self.set_controls(self.internalFrame)
        self.update_position()


# New stone dialog
class GbrAddStoneDlg(GbrChangeStoneDlg):
    def init_params(self, args, kwargs):
        GbrChangeStoneDlg.init_params(self, args, kwargs)
        self.is_new = True

    def get_title(self):
        return "Add stone"

    def select_pos_callback(self, event):
        if event.state:
            self.binder.bind(self.root.imagePanel.canvas, "<Button-1>", self.mouse_click_callback)
            self.root.imagePanel.canvas.config(cursor = 'target')
        else:
            self.binder.unbind(self.root.imagePanel.canvas, "<Button-1>")
            self.root.imagePanel.canvas.config(cursor = '')

    def mouse_click_callback(self, event):
        x, y = self.root.imagePanel.frame2image((event.x, event.y))
        stone = self.root.board.find_stone(c = (x, y))
        if stone is not None:
            self.root.statusBar.set("Stone already exists at this position")
        else:
            # Find average radius of board stones
            mean_r = int(np.mean([x[GR_R] for x in self.root.board.all_stones]))

            # Convert x,y to board position
            p = convert_xy([[x, y, mean_r]], self.root.board.results)
            if p is not None and len(p) > 0:
                self.stone = list(p[0])
                self.stone.extend([STONE_BLACK])

            # Update visuals
            self.posBtn.release()
            self.set_vars()
            self.set_controls(self.internalFrame)
            if self.stone is not None:
                self.root.show_stone(self.stone)


    def close(self):
        self.binder.unbind(self.root.imagePanel.canvas, "<Button-1>")
        GbrChangeStoneDlg.close(self)

# Stones list dialog
class GbrStonesDlg(GrDialog):
    def __init__(self, *args, **kwargs):
        GrDialog.__init__(self, *args, **kwargs)

    def init_state(self):
        self.stones = None
        self.stone_dlg = None
        self.is_selecting = False
        self.binder.register(self.root, '<Stone-Selected>', self.board_stone_selected_callback)

    def get_minsize(self):
        return (200, 300)

    def get_title(self):
        return "Stones"

    def init_frame(self, internalFrame):
        f_top = tk.Frame(internalFrame)
        f_bottom = tk.Frame(internalFrame)
        f_top.pack(side = tk.TOP, fill = tk.BOTH)
        f_bottom.pack(side = tk.BOTTOM, fill = tk.BOTH)

        self.addBtn = ImgButton(f_top,
            tag = "plus_small", tooltip = "Add a stone",
            dlg_class = GbrAddStoneDlg)
        self.addBtn.pack(side = tk.LEFT, padx = 2, pady = 2)

        self.binder.register(self.addBtn, '<Dialog-Open>', self.dlg_open_callback)
        self.binder.register(self.addBtn, '<Dialog-Close>', self.dlg_close_callback)

        self.editBtn = ImgButton(f_top,
            tag = "edit_small", tooltip = "Change stone",
            dlg_class = GbrChangeStoneDlg)
        self.editBtn.pack(side = tk.LEFT, padx = 2, pady = 2)

        self.binder.register(self.editBtn, '<Click>', self.edit_click_callback)
        self.binder.register(self.editBtn, '<Dialog-Open>', self.dlg_open_callback)
        self.binder.register(self.editBtn, '<Dialog-Close>', self.dlg_close_callback)

        ImgButton(f_top,
            tag = "clear_small", tooltip = "Reset stones properties to default",
            command = self.reset_stones_click_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        self.bg = ImgButtonGroup(f_top)
        self.bg.add_group("stone", ["plus_small", "edit_small"], BG_DEPENDENT)

        sbr = tk.Scrollbar(f_bottom)
        sbr.pack(side=tk.RIGHT, fill=tk.Y)

        self.tv = ttk.Treeview(f_bottom, column = ("p"),
            height = 10, selectmode = "browse" )
        self.tv.column("#0", width=40, minwidth=20, stretch=tk.NO)
        self.tv.column("p", width=30, minwidth=20, stretch=tk.YES)
        self.tv.heading("#0", text = "Color", anchor = tk.W)
        self.tv.heading("#1", text = "Stone", anchor = tk.W)
        treeview_sort_columns(self.tv)
        self.tv.pack(side = tk.TOP, fill=tk.BOTH, expand = True)
        self.binder.bind(self.tv, "<<TreeviewSelect>>", self.select_callback)

        self.stone_images = { STONE_BLACK: ImgButton.get_ui_image("black_small.png"),
                              STONE_WHITE: ImgButton.get_ui_image("white_small.png")}
        self.update_listbox()

        sbr.config(command = self.tv.yview)
        self.tv.config(yscrollcommand = sbr.set)

    def init_buttons(self, buttonFrame):
        f_top = tk.Frame(buttonFrame)
        f_bottom = tk.Frame(buttonFrame)
        f_top.pack(side = tk.TOP, fill = tk.BOTH)
        f_bottom.pack(side = tk.BOTTOM, fill = tk.BOTH)

        self.allVar = tk.IntVar(0)
        tk.Checkbutton(f_top, text = "Select all", variable = self.allVar,
            command = self.select_all_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        NButton(f_bottom, text = "Save SGF",
            uimage = "detect_flat.png", compound="left",
            command = self.save_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        tk.Button(f_bottom, text = "Close",
            command = self.close_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

        self.buttonFrame.configure(bd = 0, relief = tk.FLAT)

    def grab_focus(self):
        self.focus_set()

    def edit_click_callback(self, event):
        """Change stone button click callback"""
        event.cancel = self.selected_stone() is None

    def reset_stones_click_callback(self, event):
        """Reset stones button click callback"""
        self.editBtn.release()
        self.root.board.stones.reset()
        self.root.imageMarker.clear()
        self.root.addedMarker.clear()
        self.update_listbox()

        event.cancel = True

    def dlg_open_callback(self, event):
        """Stone dialog open callback"""
        self.stone_dlg = event.dlg
        if type(event.dlg) is GbrChangeStoneDlg:
            self.update_stone_dlg()

    def dlg_close_callback(self, event):
        """Stone dialog close callback"""
        if not event.ok or self.stone_dlg is None:
            self.stone_dlg = None
            return

        # Update master window
        stone = self.stone_dlg.stone
        if stone is not None:
            self.root.board.stones.add([stone])
            self.root.show_stone(stone)

        if type(self.stone_dlg) is GbrAddStoneDlg:
            # Update all stones if a new stone was added
            self.update_listbox()
            self.root.addedMarker.add_stones(
                self.root.board.stones.get_stone_list(self.root.board.stones.forced_stones()),
                f_replace = True)

        # Find a stone in a treeview
        p = format_stone_pos(stone)
        bw = stone[GR_BW]
        ids = self.tv.tag_has(p)
        if ids is not None and len(ids) > 0:
            # Update image and ensure stone is selected
            self.is_selecting = True    # to prevent selection event handling
            self.tv.see(ids[0])
            self.tv.selection_set(ids[0])
            self.tv.item(ids[0], image = self.stone_images[bw], values = (p), tags = ((p,bw)))

        self.stone_dlg = None

    def save_click_callback(self):
        """Save button click callback"""
        self.root.save_sgf_callback(self)

    def select_callback(self, event):
        """List box selection change callback"""
        if self.is_selecting:
            # Already in selection
            self.is_selecting = False
            return

        stone = self.selected_stone()
        if stone is not None:
            self.root.show_stone(stone)
            self.update_stone_dlg(stone)

    def select_all_callback(self):
        """Select all checkbox callback"""
        if self.allVar.get() > 0:
            self.root.show_all_stones()
        else:
            self.root.hide_stones()

    def board_stone_selected_callback(self, event):
        """Stone selected in main window callback"""
        ids = self.tv.tag_has(format_stone_pos(event.stone))
        if ids is not None and len(ids) > 0:
            self.is_selecting = True    # to prevent selection event handling
            self.tv.see(ids[0])
            self.tv.selection_set(ids[0])

        if self.stone_dlg is not None:
            self.update_stone_dlg(event.stone)

    def close(self):
        """Graceful way to close the dialog"""
        if self.stone_dlg is not None:
            self.stone_dlg.close()
            self.stone_dlg = None

        self.root.imageMarker.clear()
        GrDialog.close(self)

    def selected_stone(self):
        """A stone currently selected in treeview"""
        stone = None
        sel = self.tv.selection()
        if sel is not None and len(sel) > 0:
            item = self.tv.item(sel[0])
            stone = self.root.board.find_stone(s = item['tags'][0], bw = item['tags'][1])
        return stone

    def update_listbox(self):
        """Populate the grid with stones"""

        # Save tuples of stone position and color
        t = self.root.board.all_stones
        ts = sorted(t, key = lambda x: np.sqrt(x[GR_A]**2 + x[GR_B]**2))
        self.stones = [(format_stone_pos(x), x[GR_BW]) for x in ts]

        # Populate the treview
        self.tv.delete(*self.tv.get_children())
        for stone in self.stones:
            self.tv.insert("", "end", image = self.stone_images[stone[1]],
                values = (stone[0]), tags = (stone))

    def update_stone_dlg(self, stone = None):
        """ Update stone dialog to match current selection"""
        if self.stone_dlg is None:
            return
        if stone is None:
            stone = self.selected_stone()
        self.stone_dlg.set_stone(stone)

# GUI class
class GbrGUI2(tk.Tk):
    # Constructor
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, "Go board")
        self.title("Go board")
        self.minsize(300, 400)

        self.log = GrLogger(self)
        self.board = GrBoard()
        self.binder = NBinder()

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
        self.toolbarPanel = tk.Frame(self.internalFrame, bd = 1, relief = tk.RAISED)
        self.toolbarPanel.pack(side = tk.TOP, fill = tk.X, expand = False)

        ImgButton(self.toolbarPanel,
            tag = "open", tooltip = "Open image",
            command = self.open_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "edge", tooltip = "Transform image", disabled = True,
            command = self.transform_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "area", tooltip = "Define board area", disabled = True,
            command = self.set_area_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "grid", tooltip = "Set board grid", disabled = True,
            command = self.set_grid_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "detect", tooltip = "Detect stones", disabled = True,
            command = self.detect_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "stones", tooltip = "List of stones", disabled = True,
            dlg_class = GbrStonesDlg).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "params", tooltip = "Detection params", disabled = True,
            dlg_class = GbrOptionsDlg).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "save", tooltip = "Save as SGF", disabled = True,
            command = self.save_sgf_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "reset", tooltip = "Reset image", disabled = True,
            command = self.reset_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        self.bg = ImgButtonGroup(self.toolbarPanel)
        self.bg.add_group("has_file", ["edge", "area", "grid", "detect", "params"])
        self.bg.add_group("edges", ["edge", "area", "grid"], BG_DEPENDENT)
        self.bg.add_group("detected", ["stones", "save"])
        self.bg.add_group("transformed", ["reset"])

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

        # Board area
        self.boardArea = ImageMask(self.imagePanel,
            allow_change = True,
            show_mask = False,
            mode = 'area',
            mask_callback = self.area_mask_callback)
        self.boardArea.mask_color = "green"

        # Board grid
        self.boardGrid = ImageMask(self.imagePanel,
            allow_change = True,
            show_mask = False,
            mode = 'grid',
            mask_callback = self.grid_mask_callback)
        self.boardGrid.mask_color = "red"

        # Image transformer
        self.imageTransform = ImageTransform(self.imagePanel,
            callback = self.end_transform_callback)

        # Stone marker
        self.imageMarker = ImageMarker(self.imagePanel, flash = 2)

        # Forced stone marker
        self.addedMarker = ImageMarker(self.imagePanel, flash = 0)
        self.addedMarker.line_color = { "_": "red", "B": "black", "W": "black" }
        self.addedMarker.fill_color = { "_": "", "B": "black", "W": "white" }
        self.addedMarker.line_width = { "_": 2, "B": 1, "W": 1 }
        self.addedMarker.fill_stipple = { "_": "", "B": "gray50", "W": "gray50" }

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
    def open_image_callback(self, event):
        """Open button click"""
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "":
            self.load_image(fn)
        event.cancel = True

    def transform_callback(self, event):
        """Transform button click"""
        if not event.state and self.imageTransform.started:
           self.imageTransform.cancel()
        else:
            self.imageMarker.clear()
            self.imageTransform.start()

    def end_transform_callback(self, t, state):
        """Image transformation complete/cancelled"""
        self.bg.buttons['edge'].state = False
        self.bg.buttons['reset'].disabled = not state
        if state:
            self.board.image = self.imagePanel.image
            self.board.param_transform_rect = t.scaled_rect
            self.board.param_board_edges = None
            self.board.param_board_size = None
            self.boardGrid.default_mask()

    def set_area_callback(self, event):
        """Set area button click"""
        if self.board.is_gen_board: return
        if event.state:
            self.imageMarker.clear()
            self.boardArea.show()
        else:
            self.boardArea.hide()

    def set_grid_callback(self, event):
        """Set grid button click"""
        if self.board.is_gen_board: return
        if event.state:
            self.imageMarker.clear()
            self.detect_edges()
            self.boardGrid.show()
        else:
            self.boardGrid.hide()

    def detect_callback(self, event):
        """Detect button click"""
        if not self.board.is_gen_board:
            self.detect_stones()
        event.cancel = True

    def save_sgf_callback(self, event):
        """SGF save button click"""
        if self.board.results is None:
            return

        fn = filedialog.asksaveasfilename(title = "Save SGF file",
            initialfile = os.path.splitext(self.board.image_file)[0] + '.sgf',
            defaultextension = '.sgf',
            filetypes = (("SGF files","*.sgf"),("All files","*.*")))
        if fn != "":
            self.save_sgf(fn)

        event.cancel = True

    def reset_image_callback(self, event):
        """Reset button click"""
        self.bg["has_file"].release(exclude = event.tag)

        self.imageMarker.clear()
        self.imageTransform.reset()
        self.board.reset_image()

        self.imagePanel.set_image(self.board.image)
        self.boardGrid.scaled_mask = self.board.param_board_edges
        self.boardGrid.size = self.board.param_board_size \
            if self.board.param_board_size is not None else DEF_BOARD_SIZE

        self.bg.buttons['reset'].disabled = True
        event.cancel = True

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
        if self.board.is_gen_board:
            self.statusBar.set("No board image loaded")
        elif len(self.board.stones.unforced_stones()) == 0:
            self.statusBar.set("Board image has to be processed, click Detect button")
        else:
            x, y = self.imagePanel.frame2image((event.x, event.y))
            stone = self.board.find_stone(c = (x, y))
            if not stone is None:
                self.show_stone(stone)
                BoardClickEvent = namedtuple('ClickEvent', ['stone', 'x', 'y'])
                self.binder.trigger(self, '<Stone-Selected>',
                    BoardClickEvent(stone, x, y))
            else:
                self.statusBar.set("")

    def status_click_callback(self, event):
        """Status bar mouse click"""
        self.log.show()

    def area_mask_callback(self, mask):
        """Grid mask resizing finished"""
        self.board.param_area_mask = mask.scaled_mask

    def grid_mask_callback(self, mask):
        """Grid mask resizing finished"""
        self.board.param_board_edges = mask.scaled_mask

    #
    # Core functions
    #
    def load_image(self, filename):
        """Load a board image"""

        # Clean up
        self.log.clear()
        self.imageMarker.clear()
        self.bg["has_file"].release()
        self.bg["has_file"].disabled = True
        self.bg["detected"].release()
        self.bg["detected"].disabled = True

        # Load image
        self.board.load_image(filename, f_process = False, f_with_params = True)

        # Display loaded image, mask and forced stones
        self.imagePanel.set_image(self.board.image)
        self.boardArea.scaled_mask = self.board.param_area_mask
        self.boardGrid.scaled_mask = self.board.param_board_edges
        self.boardGrid.size = self.board.param_board_size \
            if self.board.param_board_size is not None else DEF_BOARD_SIZE

        self.addedMarker.add_stones(
            self.board.stones.get_stone_list(self.board.stones.added_stones()),
            f_replace = True)

        # Update status
        if self.log.errors > 0:
            self.statusBar.set("Errors during file loading, click here for the log")
        else:
            self.statusBar.set_file("File loaded", self.board.image_file)
            self.bg.buttons['reset'].disabled = not self.board.can_reset_image
            self.bg["has_file"].disabled = False

    def detect_edges(self, f_force = False):
        """Detect edges and size of currently loaded board image"""
        if self.board.param_board_edges is not None and \
            self.board.param_board_size is not None and \
            not f_force:
                # Edges/size already detected
                return

        # Process
        self.log.clear()
        self.board.detect_edges()

        if self.log.errors > 0:
           self.statusBar.set("Automatic board detection failed, click here for the log")
        else:
            self.statusBar.set("{s}x{s} board detected".format(s = self.board.board_size))
            self.boardGrid.scaled_mask = self.board.param_board_edges
            self.boardGrid.size = self.board.board_size

    def detect_stones(self, highlight = False):
        """Detect stones on currently loaded board image"""
        # Clean up
        self.log.clear()
        self.bg['edges'].release()

        # Process
        self.board.process()

        # Update status
        if self.log.errors > 0:
            self.statusBar.set("Errors during processing, click here for the log")
        else:
            self.statusBar.set("{b} black, {w} white stones on {s}x{s} board detected, click here for the log".format(
                b = len(self.board.black_stones),
                w = len(self.board.white_stones),
                s = self.board.board_size))

            self.boardGrid.scaled_mask = self.board.board_edges
            self.boardGrid.size = self.board.board_size
            self.board.save_params()

            self.bg["detected"].disabled = False
            if highlight: self.show_all_stones()

    def save_sgf(self, fn):
        """Save SGF file"""
        if not self.board.is_gen_board:
            self.board.save_sgf(fn)
            self.statusBar.set_file("Board saved to ", str(fn))

    def show_stone(self, stone):
        """Highlight one stone"""
        self.statusBar.set("{} {}".format(STONE_COLORS[stone[GR_BW]], format_stone_pos(stone)))
        self.imageMarker.add_stone(stone, f_replace = True)

    def show_all_stones(self):
        """Highlight all stones"""
        self.imageMarker.clear()
        self.imageMarker.add_stones(self.board.black_stones, STONE_BLACK)
        self.imageMarker.add_stones(self.board.white_stones, STONE_WHITE)
        self.imageMarker.show()

    def hide_stones(self):
        self.imageMarker.clear()

    #
    # Utility functions
    #


# Main function
def main():
    window = GbrGUI2()
    window.mainloop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
