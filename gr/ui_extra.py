#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     UI classes and functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------
import sys
import os
from PIL import Image, ImageTk

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    from tkinter import ttk

UI_DIR = 'ui'
PADX = 5
PADY = 5

# Image frame with additional tag - for debug info
class NLabel(tk.Label):
    def __init__(self, master, tag=None, *args, **kwargs):
        tk.Label.__init__(self, master, *args, **kwargs)
        self.master, self.tag = master, tag

# ImageButton
class ImgButton(tk.Label):
    def __init__(self, master, tag, state, callback, *args, **kwargs):
        tk.Label.__init__(self, master, *args, **kwargs)

        self._tag = tag
        self._state = state
        self._callback = callback

        # Load button images
        self._images = [ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self._tag + '_up.png'))),
                        ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self._tag + '_down.png')))]

        # Update kwargs
        w = self._images[0].width() + 6
        self.configure(borderwidth = 1, relief = "groove", width = w)
        self.configure(image = self._images[self._state])

        self.bind("<Button-1>", self.mouse_click)

    def mouse_click(self, event):
        new_state = not self._state
        self.configure(image = self._images[new_state])
        if self._callback(event = event, tag = self._tag, state = new_state):
           self._state = new_state
        else:
           self.configure(image = self._images[self._state])

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        self.configure(image = self._images[new_state])

# Tooltip class
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# Add a toolip
def createToolTip(widget, text):
    """Creates a tooltip with given text for given widget"""
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

# Add a panel with caption and buttons
# Buttons is a list of lists with ImgButon parameters:
#   [0] tag
#   [1] initial state
#   [2] button callback
#   [3] (optional) tooltip
# Returns created panel, body frame/image and dictionary with buttons
def add_panel(parent, caption, btn_params, image = None, frame_callback = None):
    """Add a panel with caption and buttons"""
    # Panel itself
    panel = tk.Frame(parent)
    panel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True, padx = PADX, pady = PADY)

    # Header
    header = tk.Frame(panel)
    #header.pack(side = tk.TOP, fill = tk.X, expand = True, anchor = tk.N)
    header.grid(row = 0, column = 0, padx = PADX, pady = PADY, sticky = "nwe")

    # Header label
    label = tk.Label(header, text = caption)
    label.pack(side = tk.LEFT, fill = tk.X, expand = True)

    # Header buttons
    buttons = dict()
    for b in btn_params:
        btn = ImgButton(header, b[0], b[1], b[2])
        if len(b) > 3:
            createToolTip(btn, b[3])
        buttons[b[0]] = btn
        btn.pack(side = tk.RIGHT)

    # Body
    if image is None:
       body = tk.Frame(panel)
    else:
       body = tk.Label(panel, image = image)

    #body.pack(fill = tk.BOTH, expand = True)
    body.grid(row = 1, column = 0, padx = PADX, pady = PADY, sticky = "nswe")
    if not frame_callback is None:
        body.bind('<Button-1>', frame_callback)

    return panel, body, buttons

def treeview_sort_columns(tv):
    """Set up Treeview tv for column sorting"""

    def _sort(tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)

        # rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        # reverse sort next time
        tv.heading(col, command = lambda: _sort(tv, col, not reverse))

    for col in range(3):
        tv.heading(col, command = lambda: _sort(tv, col, False))


