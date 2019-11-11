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
import os
from PIL import Image, ImageTk
from pathlib import Path
import random
import numpy as np
import cv2
from imutils.perspective import four_point_transform

from .grdef import *
from .utils import img_to_imgtk, resize3, board_spacing, is_on_w

import sys
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
    def __init__(self, master, *args, **kwargs):
        """Creates new ImgButton.

        Parameters:
            master     Tk windows/frame
            tag        Button tag. Files names "<tag>_down.png" and "<tag>_up.png" must exist in UI_DIR.
            state      Initial pressing state (true/false)
            disabled   True/False
            tooltip    A tooltip text
            callback   Callback function. Function signature:
                          event - Tk event
                          tag   - button's tag
                          state - target state (true/false)
                       Function shall return if new state accepted, or false otherwise
        """
        self.__tag = kwargs.pop('tag', None)
        if self.__tag is None:
            raise Exception('tag not provided')
        self.__state = kwargs.pop('state', False)
        self.__disabled = kwargs.pop('disabled', False)
        self.__callback = kwargs.pop('callback', None)
        tooltip = kwargs.pop('tooltip', None)
        self.__DS_MAP = ['normal', 'disabled']

        tk.Label.__init__(self, master, *args, **kwargs)

        if not tooltip is None:
            self.__tooltip = createToolTip(self, tooltip)

        # Load button images
        self.__images = [ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self.__tag + '_up.png'))),
                        ImageTk.PhotoImage(Image.open(os.path.join(UI_DIR, self.__tag + '_down.png')))]

        # Update kwargs
        w = self.__images[0].width() + 4
        h = self.__images[0].height() + 4
        self.configure(borderwidth = 1, relief = "groove", width = w, height = h)
        self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

        self.bind("<Button-1>", self.mouse_click)

    def mouse_click(self, event):
        cur_state = self.__state
        new_state = not self.__state
        self.configure(image = self.__images[new_state], state = self.__DS_MAP[self.__disabled])
        if new_state: self._root().update()
        self.__state = new_state

        if not self.__callback is None:
           if not self.__callback(event = event, tag = self.__tag, state = new_state):
              self.__state = cur_state
              if new_state:
                 # Unpress after small delay
                 self.after(300, lambda: self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled]))
              else:
                 self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, new_state):
        if new_state != self.__state:
            self.__state = new_state
            self.configure(image = self.__images[new_state], state = self.__DS_MAP[self.__disabled])

    @property
    def disabled(self):
        return self.__disabled

    @state.setter
    def disabled(self, ds):
        if ds != self.__disabled:
            self.__disabled = ds
            self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

# Tooltip
class ToolTip(object):
    """ToolTip class (see https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter)"""

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text, c = None):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        bb = self.widget.bbox("insert")
        if bb is None: bb = self.widget.bbox("all")
        x, y, cx, cy = bb
        x = x + self.widget.winfo_rootx() + 27
        y = y + cy + self.widget.winfo_rooty() + 27
        if not c is None:
           x, y = c
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
def createToolTip(widget, text, coord = None):
    """Creates a tooltip with given text for given widget"""
    toolTip = ToolTip(widget)
    widget.bind('<Enter>', lambda f: toolTip.showtip(text, coord))
    widget.bind('<Leave>', lambda f: toolTip.hidetip())
    return toolTip

# Remove a tooltip
def removeToolTip(toolTip):
    toolTip.hidetip()
    toolTip.widget.unbind('<Enter>')
    toolTip.widget.unbind('<Leave>')

class StatusPanel(tk.Frame):
    """Status panel class"""
    def __init__(self, master, max_width, *args, **kwargs):
        """Creates StatusPanel instance. A StatusPanel is a frame with additional methods.

           Parameters:
               master          master frame
               max_width       maximal text width
        """
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
        """Set status as text. If text is larger than current panel size, it's been
        truncated from the end, ... added"""
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
        """Set text + file name as status. If file name is too long, it's been
        shrinken by eliminating some path parts in the middle"""
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

class NBinder(object):
    """ Supplementary class to manage widget bindings"""

    # Tkinter bind/unbind seems not working
    # This variable will keep all bindings set before
    __bindings = dict()

    def __init__(self):
        self.bnd_ref = dict()

    def bind(self, widget, event, callback, add = ''):
        """Bind a callback to widget event and keep the reference"""

        # Store binding in this instance
        key = str(widget.winfo_id()) + '__' + str(event)
        bnd_id = widget.bind(event, callback, add)
        self.bnd_ref[key] = [bnd_id, widget, event]

        # Store global binding
        ref = []
        if key in NBinder.__bindings: ref = NBinder.__bindings[key]
        ref.append([self, widget, event, callback])
        NBinder.__bindings[key] = ref

    def unbind(self, widget, event):
        """Unbind all callbacks for given event from a widget"""

        key = str(widget.winfo_id()) + '__' + str(event)
        if key in self.bnd_ref:
           # Unbind from this instance
           bnd_id = self.bnd_ref[key][0]
           widget.unbind(event, bnd_id)
           del self.bnd_ref[key]

           # Unbind/rebind globally
           NBinder.__unbind(self, key)

    def unbind_all(self):
        """Unbind all callbacks"""

        # Unbind all in this instance
        for key in self.bnd_ref.keys():
            bnd_id, widget, event = self.bnd_ref[key]
            widget.unbind(event, bnd_id)
            NBinder.__unbind(self, key)

        self.bnd_ref = dict()

    @staticmethod
    def __unbind(owner, key):
        if not key in NBinder.__bindings:
            return
        refs = NBinder.__bindings[key]
        for i in range(len(refs)):
            w = refs[i][0]
            if w == owner:
               # This is my binding, remove it and rebind everything remaining
               refs.pop(i)
               for ref2 in refs:
                   w2, widget, event, callback = ref2
                   widget.bind(event, callback)



# Image panel class
class ImagePanel(tk.Frame):
    def __init__(self, master, **kwargs):
        """Creates ImagePanel instance.

           Parameters:
               master         master frame

           Keyworded parameters:
               caption          Panel caption
               btn_params       Params for ImgButtons: tag, initial state, callback function, (optional) tooltip
               image            OpenCv or PhotoImage. If none provided, an empty frame is created (default is None)
               frame_callback   Callback for panel mouse click (default is None)
               mode             Either 'clip' (default) or 'fit'.
                                If 'clip', the image is statically scaled to max_size and clipped to parent frame.
                                Scale can be changed by 'scale' property. If image is scaled down, panning is allowed.
                                If 'fit, the image is scaled dynamically (resp.min_size and max_size). Parent frame
                                should have been aligned with pack(fill="both", expand = True). No scaling/panning is allowed.
               max_size         Maximum image size. If image is larger, it will be resized down to this size (default 0).
               min_size         Minimum image size when image is dynamically resized
               resize_callback  A function to be called after resizing: f(panel, old_shape, new_shape).
               scrollbars       Boolean or tuple of booleans. If True both x and y scrollbars attached to canvas.
                                If tuple provided, it specify where scrollbars are attached (horiz, vert)
        """
        # will be assign after frame init
        self.__image = None
        self.__src_image = None
        self.__scale = [1.0, 1.0]
        self.__offset = [0, 0]
        self.__image_shape = []

        # Panel parameters
        img = kwargs.pop('image', None)
        self.__caption = kwargs.pop('caption', '')
        btn_params = kwargs.pop('btn_params', None)
        frame_callback = kwargs.pop('frame_callback', None)
        f_sb = kwargs.pop('scrollbars', (False, False))
        if not type(f_sb) is tuple: f_sb = (f_sb, f_sb)
        self.__mode = kwargs.pop('mode',"clip")
        self.__max_size = kwargs.pop('max_size', 0)
        self.__min_size = kwargs.pop('min_size',0)
        self.resize_callback = kwargs.pop('resize_callback', None)
        if self.__mode != "clip" and self.__mode != "fit":
            raise ValueError("Invalid mode ", self.__mode)

        # Init
        tk.Frame.__init__(self, master, None, **kwargs)
        self.__src_image = img
        self.__set_image(img)

        # Panel to hold everything
        internalPanel = tk.Frame(self)
        internalPanel.pack(fill = tk.BOTH, expand = True)

        # Header panel and label
        headerPanel = tk.Frame(internalPanel)
        headerPanel.pack(side = tk.TOP, fill = tk.X, expand = True)

        self.__header = tk.Label(headerPanel, text = self.__caption)
        self.__header.pack(side = tk.LEFT, fill = tk.X, expand = True)

        # Buttons
        self.__buttons = dict()
        if not btn_params is None:
            for b in btn_params:
                btn = ImgButton(headerPanel,
                    tag = b[0],
                    state = b[1],
                    callback = b[2],
                    tooltip = b[3] if len(b) > 3 else None)
                self.__buttons[b[0]] = btn
                btn.pack(side = tk.RIGHT, padx = 2, pady = 2)

        # Canvas
        canvasPanel = tk.Frame(internalPanel)
        canvasPanel.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        sz = self.max_canvas_size
        self.canvas = tk.Canvas(canvasPanel,
              width = sz[0],
              height = sz[1])
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Image on canvas
        self.__image_id = None
        if not self.__imgtk is None:
            self.__image_id = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.__imgtk)

        # Scrollbars
        if f_sb[0]:
            sbb = tk.Scrollbar(internalPanel, orient=tk.HORIZONTAL)
            sbb.pack(side=tk.BOTTOM, fill=tk.X, expand = True)
            sbb.config(command=self.canvas.xview)
            self.canvas.config(xscrollcommand=sbb.set)
        if f_sb[1]:
            sbr = tk.Scrollbar(canvasPanel)
            sbr.pack(side=tk.RIGHT, fill=tk.Y, expand = True)
            sbr.config(command=self.canvas.yview)
            self.canvas.config(yscrollcommand=sbr.set)

        # Frame click callback
        self.__binder = NBinder()
        if not frame_callback is None:
           self.__binder.bind(self.canvas, '<Button-1>', frame_callback)

        # Resize handler
        if self.__mode == "fit":
           self.__binder.bind(self.canvas, "<Configure>", self.__on_configure)

    @property
    def image(self):
        """Image adjusted to panel's area"""
        return self.__image

    @image.setter
    def image(self, img):
        self.set_image(img)

    @property
    def src_image(self):
        """Original image"""
        return self.__src_image

    @property
    def imagetk(self):
        """PhotoImage image"""
        return self.__imagetk

    @property
    def mode(self):
        """Mode"""
        return self.__mode

    @mode.setter
    def mode(self, m):
        # TODO: mode change
        self.__mode = m

    @property
    def scale(self):
        """Image scale"""
        return self.__scale

    @scale.setter
    def scale(self, scale):
        if self.__mode == "clip":
            old_shape = self.__image.shape
            self.__resize(scale = scale)
            self.__update_image()
            new_shape = self.__image.shape

            if self.resize_callback is not None:
                self.resize_callback(self, old_shape, new_shape)

    @property
    def offset(self):
        """Offset of image origin"""
        return self.__offset

    @property
    def max_size(self):
        """Maximum image size"""
        return self.__max_size

    @max_size.setter
    def max_size(self, ms):
        self.__max_size = ms
        self.__resize()
        self.__update_image()

    @property
    def min_size(self):
        """Minimum image size"""
        return self.__min_size

    @min_size.setter
    def min_size(self, ms):
        # TODO: size change
        self.__min_size = ms
        self.__resize()
        self.__update_image()

    @property
    def caption(self):
        """Panel caption text"""
        return self.__caption

    @caption.setter
    def caption(self, text):
        self.__caption = text
        self.__header.configure(text = text)

    @property
    def buttons(self):
        """Image buttons shown on panel"""
        return self.__buttons

    @property
    def max_canvas_size(self):
        """Maximum actual canvas size"""
        sz = self.max_size
        if sz > 0:
            return (sz, sz)
        elif not self.__image is None:
            return (self.__image.shape[CV_WIDTH], self.__image.shape[CV_HEIGTH])
        else:
            return DEF_IMG_SIZE

    @property
    def image_shape(self):
        """Actual image shape"""
        return self.__image_shape

    @property
    def scaled_shape(self):
        """Image shape as it is displayed on canvas"""
        return self.__image_shape if max(self.__scale) >= 1.0 else self.__image.shape

    def set_image(self, img):
        """Changes image. img can be either OpenCv or PhotoImage"""
        self.__src_image = img
        self.__set_image(img)
        self.__update_image()

    def image2frame(self, p):
        """Maps a point from image coordinates to frame coordinates

        Parameters:
            p  A tuple or list of (x,y) image-related coordinates

        Returns:
            Coordinates scaled to frame as (x, y) tuple
        """
        return (int(p[0] * self.scale[0]) + self.offset[0],
                int(p[1] * self.scale[1]) + self.offset[1])

    def frame2image(self, p):
        """Maps a point from frame coordinates to image coordinates

        Parameters:
            p  A tuple or list of (x,y) frame-related coordinates

        Returns:
            Coordinates scaled to image as (x, y) tuple
        """
        return (int((p[0] - self.offset[0]) / self.scale[0]),
                int((p[1] - self.offset[1]) / self.scale[1]))

    def __set_image(self, image):
        """Internal function to assign image"""
        if image is None:
            self.__image = None
            self.__image_shape = [0,0]
            self.__offset = [0, 0]
            self.__imgtk = None
        else:
            self.__image = image
            self.__image_shape = image.shape
            self.__resize()

    def __resize(self, size = None, scale = None):
        """Internal function to resize image"""
        if self.__image is None: return
        if size is None and scale is None and self.__max_size > 0:
            size = self.__max_size   # legacy

        self.__scale = [1.0, 1.0]
        self.__offset = [0, 0]
        if size is not None or scale is not None:
            c = self.winfo_rgb(self['bg'])
            r, g, b = c[0]/256, c[1]/256, c[2]/256
            orig_shape = self.__image.shape
            self.__image, self.__scale, self.__offset = resize3(self.__src_image,
                          max_size = size,
                          scale = scale,
                          f_upsize = False,
                          f_center = True,
                          pad_color = (r, g, b))
            #print('{} -> {} x {} + {}'.format(orig_shape, self.__image.shape, self.__scale, self.__offset))
            self.__imgtk = img_to_imgtk(self.__image)

    def __update_image(self):
        """Internal function to update image"""
        if self.__imgtk is None and self.__image_id is not None:
            self.canvas.destroy(self.__image_id)
        elif not self.__imgtk is None and self.__image_id is None:
            self.__image_id = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.__imgtk)
        else:
            self.canvas.itemconfig(self.__image_id, image = self.__imgtk)

    def __on_configure(self, event):
        """ Event handler for resize events"""
        m = min(event.width, event.height)
        if self.__mode == "fit" and m < self.__max_size and m > self.__min_size:
            old_shape = self.__image.shape
            self.__resize(size = m)
            self.__update_image()
            new_shape = self.__image.shape

            if self.resize_callback is not None:
                self.resize_callback(self, old_shape, new_shape)

def addImagePanel(master, **kwargs):
    """Creates a panel with caption and buttons. Provided for backward compatibility.

    Returns:
        panel          A panel frame
    """
    return ImagePanel(master, **kwargs)

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

# Binder
# Mask class
class ImageMask(object):
    """Support class for drawing and changing mask on an image drawn on canvas"""

    def __init__(self, panel, **kwargs):
        """Creates a mask object.

        Parameters:
            panel         ImagePanel reference
            mode          Either 'area' to use to set board area mask or 'grid' to set board grid. Default is 'area'
            size          Board size for 'grid' mode (default is grdef.DEF_BOARD_SIZE)
            mask          Initial mask (if None, default mask is generated)
            allow_change  True to allow mask reshaping by user
            show_mask     True to show mask initially
            mask_callback A function to be called when mask has changed

        Mask shading and colors can be set by shade_fill, shade_stipple, mask_color attributes
        Resulting mask can be obtained through mask or scaled_mask attributes
        """
        # Parameters
        self.__panel = panel
        self.__mode = kwargs.pop('mode', 'area').lower()
        if self.__mode != 'area' and self.__mode != 'grid':
           raise ValueError('Invalid mode', self.__mode)
        self.__size = kwargs.pop('size', DEF_BOARD_SIZE)
        self.__mask = kwargs.pop('mask', None)
        self.__allow_change = kwargs.pop('allow_change', True)
        f_show = kwargs.pop('show_mask', False)
        self.__callback = kwargs.pop('mask_callback', None)

        if self.__mask is None:
            self.default_mask()

        self.__panel.resize_callback = self.__on_panel_resize

        # Public parameters
        self.shade_fill = "gray"
        self.shade_stipple = "gray50"
        self.mask_color = "red"
        self.mask_width = 2

        # Internal parameters
        self.__mask_area = None
        self.__mask_rect = None
        self.__last_cursor = None
        self.__drag_side = None

        # Draw initial mask
        if f_show: self.show()

        # Set handlers
        self.__bindings = NBinder()
        if self.__allow_change:
            self.__bindings.bind(self.__panel.canvas, "<Motion>", self.motion_callback)
            self.__bindings.bind(self.__panel.canvas, '<B1-Motion>', self.drag_callback)
            self.__bindings.bind(self.__panel.canvas, '<B1-ButtonRelease>', self.end_drag_callback)

    @property
    def panel(self):
        """ImagePanel"""
        return self.__panel

    @property
    def canvas(self):
        """Canvas"""
        return self.__panel.canvas

    @property
    def mask(self):
        """Mask as it is displayed on canvas.
           Note that internally mask can be a double due to rounding, but this
           property always return integer"""
        return np.array(self.__mask).astype('int').tolist() if self.__mask is not None else None

    @mask.setter
    def mask(self, m):
        """Mask as it is displayed on canvas"""
        self.hide()
        self.__mask = list(m)

    @property
    def scaled_mask(self):
        """Mask scaled to actual image size"""
        if self.__mask is None:
           return None
        m = self.__mask.copy()
        m[0] = int(m[0] / self.__panel.scale[0])
        m[1] = int(m[1] / self.__panel.scale[1])
        m[2] = int(m[2] / self.__panel.scale[0])
        m[3] = int(m[3] / self.__panel.scale[1])
        return m

    @scaled_mask.setter
    def scaled_mask(self, mask):
        """Mask scaled to actual image size"""
        if mask is None:
           self.__mask = None
        else:
           m = mask.copy()
           m[0] = int(m[0] * self.__panel.scale[0])
           m[1] = int(m[1] * self.__panel.scale[1])
           m[2] = int(m[2] * self.__panel.scale[0])
           m[3] = int(m[3] * self.__panel.scale[1])
           self.__mask = m

    @property
    def allow_change(self):
        return self.__allow_change

    @allow_change.setter
    def allow_change(self, f):
        self.__allow_change = f
        if self.__allow_change:
            self.__bindings.bind(self.__panel.canvas, "<Motion>", self.motion_callback)
            self.__bindings.bind(self.__panel.canvas, '<B1-Motion>', self.drag_callback)
            self.__bindings.bind(self.__panel.canvas, '<B1-ButtonRelease>', self.end_drag_callback)
        else:
            self.__bindings.unbind_all()

    @property
    def mode(self):
        """ImageMask mode ('area', 'grid')"""
        return self.__mode

    @mode.setter
    def mode(self, m):
        """ImageMask mode ('area', 'grid')"""
        m_ = m.lower()
        if m_ != 'area' and m_ != 'grid':
           raise ValueError('Invalid mode', m_)
        self.hide()
        self.__mode = m_

    @property
    def size(self):
        """Board size"""
        return self.__size

    @size.setter
    def size(self, sz):
        """Board size"""
        self.hide()
        self.__size = sz

    def motion_callback(self, event):
        """Callback for mouse move event"""
        CURSORS = ["left_side", "top_side", "right_side", "bottom_side"]
        c = None
        if not self.__mask_rect is None:
            side = self.__get_mask_rect_side(event.x, event.y)
            if not side is None: c = CURSORS[side]

        if c is None and not self.__last_cursor is None:
            # Left rectangle, set cursor to default
            self.canvas.config(cursor='')
            self.__last_cursor = None
        elif not c is None and self.__last_cursor != c:
            # On a line, set a cursor
            self.canvas.config(cursor=c)
            self.__last_cursor = c

    def drag_callback(self, event):
        """Callback for mouse drag event"""
        if self.__drag_side is None:
            self.__drag_side = self.__get_mask_rect_side(event.x, event.y)
        if not self.__drag_side is None:
            # Calculate new coordinates
            p = (self.canvas.canvasx(event.x) - self.__panel.offset[0],
                 self.canvas.canvasy(event.y) - self.__panel.offset[1])
            if self.__drag_side == 0:
                self.__mask[0] = max(p[0], 0)
            elif self.__drag_side == 1:
                self.__mask[1] = max(p[1], 0)
            elif self.__drag_side == 2:
                self.__mask[2] = min(p[0], self.__panel.scaled_shape[1])
            elif self.__drag_side == 3:
                self.__mask[3] = min(p[1], self.__panel.scaled_shape[0])

            # Reposition mask rect
            m = self.mask
            self.canvas.coords(self.__mask_rect,
                 m[0] + self.__panel.offset[0] + self.mask_width,
                 m[1] + self.__panel.offset[1] + self.mask_width,
                 m[2] + self.__panel.offset[0],
                 m[3] + self.__panel.offset[1])

            # Draw shading or grid
            if self.__mode == 'area':
                self.__draw_mask_shading()
            else:
                self.__draw_mask_grid()

    def end_drag_callback(self, event):
        """Callback for mouse button release event"""
        if not self.__drag_side is None and not self.__callback is None:
           self.__callback(self)
        self.__drag_side = None

    def show(self):
        """Draw a mask or grid on canvas"""
        if self.__mask is None: self.default_mask()
        if self.__mode == 'area':
            self.__draw_mask_shading()
        else:
            self.__draw_mask_grid()
        self.__draw_mask_rect()

    def hide(self):
        """Hide a previously shown mask or grid"""
        if not self.__mask_area is None:
            for m in self.__mask_area:
                self.canvas.delete(m)
            self.__mask_area = None
        if not self.__mask_rect is None:
            self.canvas.delete(self.__mask_rect)
            self.__mask_rect = None

    def is_shown(self):
        """ True if mask is currently shown"""
        return self.__mask_area is not None or self.__mask_rect is not None

    def random_mask(self):
        """Generates a random mask"""
        dx = 0
        dy = 0
        cx = int(self.__panel.scaled_shape[CV_WIDTH] / 2) - dx
        cy = int(self.__panel.scaled_shape[CV_HEIGTH] / 2) - dy
        mx = self.__panel.scaled_shape[CV_WIDTH] - 2*dx
        my = self.__panel.scaled_shape[CV_HEIGTH] - 2*dy
        self.__mask = [
                random.randint(dx, cx),
                random.randint(dy, cy),
                random.randint(cx+1, mx),
                random.randint(cy+1, my)]

    def default_mask(self):
        """Generates default mask"""
        dx = 0
        dy = 0
        self.__mask = [
                dx,
                dy,
                self.__panel.scaled_shape[CV_WIDTH] - 2*dx,
                self.__panel.scaled_shape[CV_HEIGTH] - 2*dy]

    def __get_mask_rect_side(self, x, y):
        """Internal function. Returns a side where the cursor is on or None"""
        if self.__mask_rect is None:
            return None

        p = (self.canvas.canvasx(x), self.canvas.canvasy(y))
        b = self.canvas.coords(self.__mask_rect)

        side = None
        if is_on_w((b[0], b[1]), (b[0], b[3]), p):
            side = 0
        elif is_on_w((b[0], b[1]), (b[2], b[1]), p):
            side = 1
        elif is_on_w((b[2], b[1]), (b[2], b[3]), p):
            side = 2
        elif is_on_w((b[0], b[3]), (b[2], b[3]), p):
            side = 3

        return side

    def __draw_mask_shading(self):
        """Internal function. Draw a shading part of mask"""
        def _rect(points):
            return self.canvas.create_polygon(
                  *points,
                  outline = "",
                  fill = self.shade_fill,
                  stipple = self.shade_stipple)

        # Clean up
        if not self.__mask_area is None:
            for m in self.__mask_area:
                self.canvas.delete(m)
            self.__mask_area = None

        # Create mask points array
        m = self.mask
        sx = self.__panel.offset[0]
        sy = self.__panel.offset[1]
        ix = sx + self.__panel.scaled_shape[1]
        iy = sy + self.__panel.scaled_shape[0]
        mx = sx + m[0]
        my = sy + m[1]
        wx = sx + m[2]
        wy = sy + m[3]
        if sx == 0: sx += self.mask_width
        if sy == 0: sy += self.mask_width

        self.__mask_area = [
          _rect([sx, sy, ix, sy, ix, my, sx, my, sx, sy]),
          _rect([sx, my, mx, my, mx, iy, sx, iy, sx, my]),
          _rect([mx, wy, ix, wy, ix, iy, mx, iy, mx, wy]),
          _rect([wx, my, ix, my, ix, wy, wx, wy, wx, my])
        ]

    def __draw_mask_rect(self):
        """Internal function. Draws a mask rectangle"""
        # Clean up
        if not self.__mask_rect is None:
            self.canvas.delete(self.__mask_rect)
            self.__mask_rect = None

        # Draw rect
        m = self.mask
        sx = self.__panel.offset[0]
        sy = self.__panel.offset[1]
        mx = sx + m[0] + self.mask_width
        my = sy + m[1] + self.mask_width
        wx = sx + m[2]
        wy = sy + m[3]

        self.__mask_rect = self.canvas.create_rectangle(
          mx, my,
          wx, wy,
          outline = self.mask_color,
          width = self.mask_width
        )

    def __draw_mask_grid(self):
        """Internal function. Draws a grid"""

        # Clean up
        if not self.__mask_area is None:
            for m in self.__mask_area:
                self.canvas.delete(m)
            self.__mask_area = None

        # Params
        m = self.mask
        sx = self.__panel.offset[0]
        sy = self.__panel.offset[1]
        mx = sx + m[0] + self.mask_width
        my = sy + m[1] + self.mask_width
        wx = sx + m[2]
        wy = sy + m[3]
        space_x, space_y = board_spacing( ((mx, my),(wx,wy)), self.__size)

        # Draw V-lines
        # Actual rectangle is drawn by __draw_rect, so skip 1st and last lines
        self.__mask_area = []
        for i in range(self.__size-2):
            x1 = int(mx + ((i+1) * space_x))
            y1 = int(my)
            x2 = x1
            y2 = int(wy)
            m = self.canvas.create_line(x1, y1, x2, y2, fill = self.mask_color, width = self.mask_width)
            self.__mask_area.append(m)

        # Draw H-lines
        # Actual rectangle is drawn by __draw_rect, so skip 1st and last lines
        for i in range(self.__size-2):
            x1 = int(mx)
            y1 = int(my + ((i+1) * space_y))
            x2 = int(wx)
            y2 = y1
            m = self.canvas.create_line(x1, y1, x2, y2, fill = self.mask_color, width = self.mask_width)
            self.__mask_area.append(m)

    def __on_panel_resize(self, panel, old_shape, new_shape):
        """Callback for panel resize (internal function)"""
        m = self.__mask.copy()
        m[0] = m[0] / old_shape[CV_WIDTH] * new_shape[CV_WIDTH]
        m[1] = m[1] / old_shape[CV_HEIGTH] * new_shape[CV_HEIGTH]
        m[2] = m[2] / old_shape[CV_WIDTH] * new_shape[CV_WIDTH]
        m[3] = m[3] / old_shape[CV_HEIGTH] * new_shape[CV_HEIGTH]
        self.__mask = m
        if self.is_shown: self.show()


# Image transformer
class ImageTransform(object):
    def __init__(self, panel, callback = None):
        """Create ImageTransorm instance

            Parameters:
                panel     An ImagePanel object with Image displayed on
                callback  A callback function to be called upon transform completed or cancelled
                          Function signature: f(transformer, state) where
                            tranformer ImageTranform object
                            state      True if transformation completed or False if cancelled
        """
        self.__panel = panel
        self.__transform_state = False
        self.__tranform_rect = None
        self.__transform_help = None
        self.__transform_scale = None
        self.__transform_offset = None
        self.__src_image = None
        self.__callback = callback
        self.__bindings = NBinder()

        # Public properties
        self.show_coord = False

    @property
    def started(self):
        """True if transformation is running"""
        return self.__transform_state

    @property
    def image(self):
        """Source image"""
        return self.__panel.image

    @property
    def src_image(self):
        """Image before transformation. Assigned when transformation starts"""
        return self.__src_image

    @property
    def transform_rect(self):
        """Transformation rectangle (TL, TR, BL, BR) as it is displayed on screen"""
        if self.__tranform_rect is None:
           return None
        else:
           t = [t[:2].view(int).tolist() for t in self.__tranform_rect]
           return t

    @property
    def scaled_rect(self):
        """Transformation rectangle (TL, TR, BL, BR) scaled to actual image size"""
        t = self.transform_rect
        if t is None:
           return None
        else:
           t2 = []
           for i in t:
               t2.append([
                    int(i[0] / self.__transform_scale[0]) - self.__transform_offset[0],
                    int(i[1] / self.__transform_scale[1]) - self.__transform_offset[1] ])
           print('{} -> {}'.format(t, t2))
           return t2

    @property
    def callback(self):
        """A callback function"""
        return self.__callback

    @callback.setter
    def callback(self, c):
        """A callback function"""
        self.__callback = callback

    def start(self):
        """Initiates a transform operation"""
        if self.__transform_state:
           self.cancel()
           return False

        # Clean up
        self.__clean_up()

        # Register bindings, display help message
        self.__bindings.bind(self.__panel.canvas, "<Button-1>", self.mouse_callback)
        self.__bindings.bind(self.__panel.winfo_toplevel(), "<Escape>", self.key_callback)

        bb_help = self.__panel.canvas.bbox(tk.ALL)
        cx = int((bb_help[0] + bb_help[2])/2)
        cy = int((bb_help[1] + bb_help[3])/2)
        self.__transform_help = createToolTip(self.__panel.canvas,
            'Click on 4 image corners or press ESC to cancel', (cx, cy))

        # Initiate user actions
        self.__transform_state = True
        self.__src_image = self.__panel.src_image
        self.__panel.canvas.after(100, self.check_transform_state)
        return True

    def cancel(self):
        """Cancel transformation which was already started"""
        self.__clean_up()
        if not self.__callback is None:
           self.__callback(self, False)

    def reset(self):
        """Reset to source image"""
        if self.__src_image is None: return
        if self.__transform_state: self.cancel()
        self.__panel.image = self.__src_image

    def mouse_callback(self, event):
        """Mouse callback"""
        def show_click(n):
            circle_id = self.__panel.canvas.create_oval(event.x-4, event.y-4,
                event.x+4, event.y+4, fill="red", outline = "red")
            text_id = 0
            if self.show_coord:
               text_id = self.__panel.canvas.create_text(event.x, event.y+10,
                            text = "({},{})".format(event.x, event.y), fill = "red")
            self.__tranform_rect[n,0] = event.x
            self.__tranform_rect[n,1] = event.y
            self.__tranform_rect[n,2] = circle_id
            self.__tranform_rect[n,3] = text_id

        if not self.__transform_state: return

        if self.__tranform_rect is None:
           self.__tranform_rect = np.zeros((4,4), dtype = np.uint32)
           show_click(0)
        else:
           # Count 4 clicks
           for n in range(len(self.__tranform_rect)):
               if self.__tranform_rect[n][2] == 0:
                  show_click(n)
                  self.__transform_state = (n < len(self.__tranform_rect)-1)
                  return
           self.__transform_state = False

    def key_callback(self, event):
        """ESC key press callback"""
        self.cancel()

    def check_transform_state(self):
        """Timer callback"""
        if self.__transform_state:
           # Still running, repeat check
           self.__panel.canvas.after(100, self.check_transform_state)
        else:
           if not self.__tranform_rect is None:
              # Save current scale and offset since they will change with image change
              self.__transform_scale = self.__panel.scale
              self.__transform_offset = self.__panel.offset

              # Do 4-points transform
              t = np.array([t[:2] for t in self.__tranform_rect])
              im = four_point_transform(self.__panel.image, t)
              self.__panel.set_image(im)

              if not self.__callback is None:
                 self.__callback(self, True)
           # Clean up
           self.__clean_up()

    def __clean_up(self):
        """Internal function to clear after transformation cancelling"""
        # Remove selection points
        self.__transform_state = False
        if self.__tranform_rect is not None:
           for i in self.__tranform_rect:
               self.__panel.canvas.delete(i[2])
               if i[3] > 0: self.__panel.canvas.delete(i[3])
           self.__tranform_rect = None

        # Remove tooltip
        if not self.__transform_help is None:
           removeToolTip(self.__transform_help)
           self.__transform_help = None

        # Cancel bindings
        self.__bindings.unbind_all()

