#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     UI classes and functions
#
# Author:      kol
#
# Created:     04.07.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
import sys
import os
from PIL import Image, ImageTk
from pathlib import Path

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
    import tkFont as font
else:
    import tkinter as tk
    from tkinter import ttk, font

UI_DIR = 'ui'    # directory containing ImgButton images
PADX = 5
PADY = 5

class NLabel(tk.Label):
    """Label with additional tag"""
    def __init__(self, master, tag=None, *args, **kwargs):
        tk.Label.__init__(self, master, *args, **kwargs)
        self.master, self.tag = master, tag

# ImageButton
class ImgButton(tk.Label):
    """Button with image face"""
    def __init__(self, master, tag, state, callback, *args, **kwargs):
        """Creates new ImgButton. Parameters:

        master     Tk windows/frame
        tag        Button tag. Files names "<tag>_down.png" and "<tag>_up.png" must exist in UI_DIR.
        state      Initial state (true/false)
        callback   Callback function. Function signature:
                      event - Tk event
                      tag   - button's tag
                      state - target state (true/false)
                   Function shall return if new state accepted, or false otherwise
        """
        tk.Label.__init__(self, master, *args, **kwargs)

        self._tag = tag
        self._state = state
        self._callback = callback

        # Load button images
        self._images = [ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self._tag + '_up.png'))),
                        ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self._tag + '_down.png')))]

        # Update kwargs
        w = self._images[0].width() + 4
        h = self._images[0].height() + 4
        self.configure(borderwidth = 1, relief = "groove", width = w, height = h)
        self.configure(image = self._images[self._state])

        self.bind("<Button-1>", self.mouse_click)

    def mouse_click(self, event):
        new_state = not self._state
        self.configure(image = self._images[new_state])
        if new_state: self._root().update()
        if self._callback(event = event, tag = self._tag, state = new_state):
           self._state = new_state
        else:
           if new_state:
              # Unpress after small delay
              self.after(300, lambda: self.configure(image = self._images[self._state]))
           else:
              self.configure(image = self._images[self._state])

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        self.configure(image = self._images[new_state])

# Tooltip
class ToolTip(object):
    """ToolTip class (see https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter)"""

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

class StatusPanel(tk.Frame):
    """Status panel class"""
    def __init__(self, master, max_width, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self._max_width = max_width
        self._var = tk.StringVar()
        self._var.set("")

        self._label = tk.Label(self, textvariable = self._var, anchor = tk.W)
        self._label.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    @property
    def status(self):
        """Status text"""
        return self._var.get()

    @status.setter
    def status(self, text):
        """Status text"""
        self.set(text)

    @property
    def max_width(self):
        """Max status panel width in pixels"""
        return self._max_width

    @max_width.setter
    def max_width(self, v):
        """Max status panel width in pixels"""
        self._max_width = v

    def _get_maxw(self):
        w = self.winfo_width()
        return w

    def set(self, text):
        """Set status as text"""
        f = font.Font(font = self._label['font'])
        chw = f.measure('W')

        maxw = self._get_maxw()
        curw = f.measure(text)
        maxw -= chw*3
        if curw > maxw:
           strip_len = int((curw - maxw) / chw) + 3
           text = text[:-strip_len] + '...'

        self._var.set(text)

    def set_file(self, text, file):
        """Set status as text + file name"""
        f = font.Font(font = self._label['font'])
        chw = f.measure('W')

        maxw = self._get_maxw()
        if maxw == 0:
           maxw = self.winfo_width()
           if maxw < 20:
              # Window not updated yet
              self._var.set(text)
              return

        maxw -= chw*3
        if f.measure(text + file) > maxw:
           # Exclude file path parts to fit in starting from 3 entry
           parts = list(Path(file).parts)
           for n in range(len(parts)-3):
               parts.pop(len(parts)-2)
               t = parts[0] + '\\'.join(parts[1:])
               if f.measure(text + t) < maxw: break
           file = parts[0] + '\\'.join(parts[1:-2])
           file += '\\...\\' + parts[-1]

        self._var.set(text + file)



def addImagePanel(parent, caption, btn_params, image = None, frame_callback = None):
    """Creates a panel with caption and buttons

    Parameters:
        parent         Tk window/frame to add panel to
        caption        Panel caption
        btn_params     Params for ImgButtons: tag, initial state, callback function, (optional) tooltip
        image          PhotoImage. If none provided, an empty frame is created
        frame_callback Callback for panel mouse click

    Returns:
        panel          A panel frame
    """
    # Panel itself
    panel = tk.Frame(parent)
    panel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    # Header
    header = tk.Frame(panel)
    #header.pack(side = tk.TOP, fill = tk.X, expand = True, anchor = tk.N)
    header.grid(row = 0, column = 0, padx = 2, pady = 2, sticky = "nwe")

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
        btn.pack(side = tk.RIGHT, padx = 2, pady = 2)

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
    """Set up Treeview tv for column sorting (see https://stackoverflow.com/questions/1966929/tk-treeview-column-sort)"""

    def _sort(tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)

        # rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        # reverse sort next time
        tv.heading(col, command = lambda: _sort(tv, col, not reverse))

    for col in range(len(tv['columns'])):
        tv.heading(col, command = lambda _col=col: _sort(tv, _col, col == 0))


def addStatusPanel(parent, max_width = 0):
    """Creates a status panel"""
    return StatusPanel(parent, max_width)

def addField(parent, type_, caption, nrow, ncol, def_val):
    """Add an text entry or combobox"""
    _var = tk.StringVar()
    if not def_val is None: _var.set(str(def_val))
    tk.Label(parent, text = caption).grid(row = nrow, column = ncol)

    _entry = None
    if type_ == 'cb':
        _entry = ttk.Combobox(parent, state="readonly", textvariable = _var)
    else:
        _entry = tk.Entry(parent, textvariable = _var)
    _entry.grid(row = nrow, column = ncol + 1)
    return _var, _entry


