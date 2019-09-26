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
import random

if sys.version_info[0] < 3:
    from grdef import *
    from utils import img_to_imgtk, resize3
else:
    from gr.grdef import *
    from gr.utils import img_to_imgtk, resize3

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
            tooltip    A tooltip text
            callback   Callback function. Function signature:
                          event - Tk event
                          tag   - button's tag
                          state - target state (true/false)
                       Function shall return if new state accepted, or false otherwise
        """
        self._tag = kwargs.pop('tag', None)
        if self._tag is None:
            raise Exception('tag not provided')
        self._state = kwargs.pop('state', False)
        self._callback = kwargs.pop('callback', None)
        tooltip = kwargs.pop('tooltip', None)

        tk.Label.__init__(self, master, *args, **kwargs)

        if not tooltip is None:
            self._tooltip = createToolTip(self, tooltip)

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

# Image panel class
class ImagePanel(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        """Creates ImagePanel instance.

           Allowed keyworded parameters:
               master         master frame
               caption        Panel caption
               btn_params     Params for ImgButtons: tag, initial state, callback function, (optional) tooltip
               image          OpenCv or PhotoImage. If none provided, an empty frame is created (default is None)
               frame_callback Callback for panel mouse click (default is None)
               max_size       maximal image size (if image is larger, it will be resized down to this size)
                              Can be used only for OpenCV images (default is 0)
               scrollbars     Boolean or tuple of booleans. If True both x and y scrollbars attached to canvas.
                              If tuple provided, it specify where scrollbars are attached (horiz, vert)
               use_mask       If True, an ImageMask is created
               mask           If use_mask = True, specifies initial mask
               allow_change   If use_mask = True, specifies whether the mask can be changed
               show_mask      If use_mask = True, specifies whether the mask should be displayed immediatelly
        """

        # Parse parameters
        self.__max_size = kwargs.pop('max_size', 0)
        img = kwargs.pop('image', None)
        self.__caption = kwargs.pop('caption', '')
        btn_params = kwargs.pop('btn_params', None)
        frame_callback = kwargs.pop('frame_callback', None)
        f_sb = kwargs.pop('scrollbars', (False, False))
        if not type(f_sb) is tuple: f_sb = (f_sb, f_sb)

        use_mask = kwargs.pop('use_mask', False)
        mask = kwargs.pop('mask', None)
        allow_change = kwargs.pop('allow_change', False)
        show_mask = kwargs.pop('show_mask', False)
        min_dist = kwargs.pop('min_dist', 0)

        # Init
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.__scale = [1.0, 1.0]
        self.__offset = [0, 0]
        self.__image_shape = []
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
        if not frame_callback is None:
            self.canvas.bind('<Button-1>', frame_callback)

        # Image mask
        self.__image_mask = None
        if use_mask:
           self.create_image_mask(mask = mask,
                allow_change = allow_change,
                show_mask = show_mask,
                min_dist = min_dist)

    @property
    def image(self):
        """OpenCv image"""
        return self.__image

    @property
    def imagetk(self):
        """PhotoImage image"""
        return self.__imagetk

    @image.setter
    def image(self, img):
        self.set_image(img)

    @imagetk.setter
    def imagetk(self, imgtk):
        self.set_image(imgtk)

    @property
    def scale(self):
        """Image scale"""
        return self.__scale

    @property
    def offset(self):
        """Offset of image origin"""
        return self.__offset

    @property
    def max_size(self):
        """Maximum panel size"""
        return self.__max_size

    @max_size.setter
    def max_size(self, ms):
        self.__max_size = ms
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
    def image_mask(self):
        """Image mask or None"""
        return self.__image_mask

    def set_image(self, img):
        """Changes image. img can be either OpenCv or PhotoImage"""
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

    def create_image_mask(self, **kwargs):
        """Creates ImageMask if it was not created upon init"""
        if self.__image is None:
           raise Exception('Cannot create ImageMask if image is not set')
        kwargs['offset'] = self.__offset
        self.__image_mask = ImageMask(self.canvas, self.__image_shape, **kwargs)

    def __set_image(self, image):
        """Internal function to assign image"""
        if image is None:
            self.__image = None
            self.__image_shape = [0,0]
            self.__imgtk = None
        elif type(image) is ImageTk.PhotoImage:
            self.__image = None
            self.__imgtk = image
        else:
            self.__image = image.copy()
            self.__image_shape = self.__image.shape
            self.__resize()
            self.__imgtk = img_to_imgtk(self.__image)

    def __resize(self):
        """Internal function to resize image"""
        self.__scale = [1.0, 1.0]
        self.__offset = [0, 0]
        if not self.__image is None and self.max_size > 0:
            c = self.winfo_rgb(self['bg'])
            r, g, b = c[0]/256, c[1]/256, c[2]/256
            self.__image, self.__scale, self.__offset = resize3(self.__image,
                          max_size = self.__max_size,
                          f_upsize = False,
                          f_center = True,
                          pad_color = (r, g, b))
            #print('Shape: {}, scale: {}, offset: {}'.format(orig_shape, self.__scale, self.__offset))
            try:
               if not self.__image_mask is None:
                  self.__image_mask.set_shape(self.__image_shape, self.__offset)
            except AttributeError:
                  pass

    def __update_image(self):
        """Internal function to update image"""
        if self.__imgtk is None and self.__image_id is not None:
            self.canvas.destroy(self.__image_id)
        elif not self.__imgtk is None and self.__image_id is None:
            self.__image_id = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.__imgtk)
        else:
            self.canvas.itemconfig(self.__image_id, image = self.__imgtk)


def addImagePanel(parent, **kwargs):
    """Creates a panel with caption and buttons

       Allowed parameters:
           master         master frame
           caption        Panel caption
           btn_params     Params for ImgButtons: tag, initial state, callback function, (optional) tooltip
           image          OpenCv or PhotoImage. If none provided, an empty frame is created
           frame_callback Callback for panel mouse click
           max_size       maximal image size (if image is larger, it will be resized down to this size)
                          Can be used only for OpenCV images.

    Returns:
        panel          A panel frame
    """
    return ImagePanel(parent, None, **kwargs)

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

def is_on(a, b, c):
    """Return true if point c intersects the line segment from a to b."""

    def collinear(a, b, c):
        "Return true iff a, b, and c all lie on the same line."
        return (b[0] - a[0]) * (c[1] - a[1]) == (c[0] - a[0]) * (b[1] - a[1])

    def within(p, q, r):
        "Return true iff q is between p and r (inclusive)."
        return p <= q <= r or r <= q <= p

    # (or the degenerate case that all 3 points are coincident)
    return (collinear(a, b, c)
            and (within(a[0], c[0], b[0]) if a[0] != b[0] else
                 within(a[1], c[1], b[1])))

def is_on_w(a,b,c,delta=1):
    """Return true if point c intersects the line segment from a to b. Delta specifies a gap"""
    for i in range(delta*3):
        x = c[0] + i - 1
        for j in range(delta*3):
            y = c[1] + j - 1
            if is_on(a, b, (x, y)): return True
    return False


# Mask class
class ImageMask(object):
    """Support class for drawing and changing mask on an image drawn on canvas"""

    def __init__(self, canvas, image_shape, **kwargs):
        """Creates a mask object.

        Parameters:
            canvas        A canvas object (required)
            image_shape   Shape of image drawn on canvas- tuple (height, width)
            offset        Offset of image on canvas ([x, y])
            mask          Initial mask (if None, default mask is generated)
            allow_change  True to allow mask reshaping by user (dragging by mouse)
            show_mask     True to show mask initially
            min_dist      Minimal allowed distance to edges

        Mask shading and colors can be set by shade_fill, shade_stipple, mask_color attributes
        Resulting mask can be obtained through mask attribute
        """
        # Parameters
        self.__canvas = canvas
        self.__image_shape = image_shape
        self.__offset = kwargs.pop('offset', [0,0])
        self.__mask = kwargs.pop('mask', None)
        self.__allow_change = kwargs.pop('allow_change', True)
        self.__min_dist = kwargs.pop('min_dist', 0)
        f_show = kwargs.pop('show_mask', True)

        if self.mask is None:
            self.default_mask()

        # Public parameters
        self.shade_fill = "gray"
        self.shade_stipple = "gray50"
        self.mask_color = "red"

        # Internal parameters - should not be changed
        self.mask_area = None
        self.mask_rect = None
        self.last_cursor = None
        self.drag_side = None

        # Draw initial mask
        if f_show: self.show()

        # Set handlers if required
        if self.__allow_change:
            self.canvas.bind("<Motion>", self.motion_callback)
            self.canvas.bind('<B1-Motion>', self.drag_callback)
            self.canvas.bind('<B1-ButtonRelease>', self.end_drag_callback)

    @property
    def canvas(self):
        return self.__canvas

    @property
    def image_shape(self):
        return self.__image_shape

    @image_shape.setter
    def image_shape(self, shape):
        was_shown = not self.mask_rect is None
        self.hide()
        self.__image_shape = shape
        if was_shown: show()

    @property
    def offset(self):
        return self.__offset

    @offset.setter
    def offset(self, ofs):
        was_shown = not self.mask_rect is None
        self.hide()
        self.__offset = ofs
        if was_shown: show()

    @property
    def mask(self):
        return self.__mask

    @mask.setter
    def mask(self, m):
        was_shown = not self.mask_rect is None
        self.hide()
        self.__mask = m
        if was_shown: show()

    @property
    def allow_change(self):
        return self.__allow_change

    @allow_change.setter
    def allow_change(self, f):
        self.__allow_change = f
        if self.__allow_change:
            self.canvas.bind("<Motion>", self.motion_callback)
            self.canvas.bind('<B1-Motion>', self.drag_callback)
            self.canvas.bind('<B1-ButtonRelease>', self.end_drag_callback)
        else:
            self.canvas.unbind("<Motion>")
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<B1-ButtonRelease>')

    @property
    def min_dist(self):
        """Minimal distance to edges"""
        return self.__min_dist

    @min_dist.setter
    def min_dist(self, m):
        """Minimal distance to edges"""
        self.__min_dist = m

    def set_shape(self, shape, ofs = None):
        """Changes image shape and offset"""
        was_shown = not self.mask_rect is None
        self.hide()
        self.__image_shape = shape
        if not ofs is None: self.__offset = ofs
        if was_shown: show()

    def motion_callback(self, event):
        """Callback for mouse move event"""
        CURSORS = ["left_side", "top_side", "right_side", "bottom_side"]
        c = None
        if not self.mask_rect is None:
            side = self.__get_mask_rect_side(event.x, event.y)
            if not side is None: c = CURSORS[side]

        if c is None and not self.last_cursor is None:
            # Left rectangle, set cursor to default
            self.canvas.config(cursor='')
            self.last_cursor = None
        elif not c is None and self.last_cursor != c:
            # On a line, set a cursor
            self.canvas.config(cursor=c)
            self.last_cursor = c

    def drag_callback(self, event):
        """Callback for mouse drag event"""
        if self.drag_side is None:
            self.drag_side = self.__get_mask_rect_side(event.x, event.y)
        if not self.drag_side is None:
            p = (self.canvas.canvasx(event.x) - self.offset[0],
                 self.canvas.canvasy(event.y) - self.offset[1])
            if self.drag_side == 0:
                self.mask[0] = max(p[0], self.min_dist)
            elif self.drag_side == 1:
                self.mask[1] = max(p[1], self.min_dist)
            elif self.drag_side == 2:
                self.mask[2] = min(p[0], self.image_shape[1]-self.min_dist)
            elif self.drag_side == 3:
                self.mask[3] = min(p[1], self.image_shape[0]-self.min_dist)

            self.canvas.coords(self.mask_rect,
                 self.mask[0] + self.offset[0],
                 self.mask[1] + self.offset[1],
                 self.mask[2] + self.offset[0],
                 self.mask[3] + self.offset[1])

            self.__draw_mask_shading()

    def end_drag_callback(self, event):
        """Callback for mouse button release event"""
        #print('Drag end')
        self.drag_side = None

    def show(self):
        """Draw a mask on canvas"""
        self.__draw_mask_shading()
        self.__draw_mask_rect()

    def hide(self):
        """Hide a previously shown mask"""
        if not self.mask_area is None:
            for m in self.mask_area:
                self.canvas.delete(m)
            self.mask_area = None
        if not self.mask_rect is None:
            self.canvas.delete(self.mask_rect)
            self.mask_rect = None

    def random_mask(self):
        """Generates a random mask"""
        dx = self.min_dist
        dy = self.min_dist
        cx = int(self.image_shape[CV_WIDTH] / 2) - dx
        cy = int(self.image_shape[CV_HEIGTH] / 2) - dy
        mx = self.image_shape[CV_WIDTH] - 2*dx
        my = self.image_shape[CV_HEIGTH] - 2*dy
        self.__mask = [
                random.randint(dx, cx),
                random.randint(dy, cy),
                random.randint(cx+1, mx),
                random.randint(cy+1, my)]

    def default_mask(self):
        """Generates default mask"""
        dx = self.min_dist
        dy = self.min_dist
        self.__mask = [
                dx,
                dy,
                self.image_shape[CV_WIDTH] - 2*dx,
                self.image_shape[CV_HEIGTH] - 2*dy]


    def __draw_mask_shading(self):
        """Internal function. Draw a shading part of mask"""
        def _rect(points):
            return self.canvas.create_polygon(
                  *points,
                  outline = "",
                  fill = self.shade_fill,
                  stipple = self.shade_stipple)

        # Clean up
        if not self.mask_area is None:
            for m in self.mask_area:
                self.canvas.delete(m)
            self.mask_area = None

        # Create mask points array
        sx = self.offset[0]
        sy = self.offset[1]
        ix = sx + self.image_shape[1]
        iy = sy + self.image_shape[0]
        mx = sx + self.mask[0]
        my = sy + self.mask[1]
        wx = sx + self.mask[2]
        wy = sy + self.mask[3]
        self.mask_area = [
          _rect([sx, sy, ix, sy, ix, my, sx, my, sx, sy]),
          _rect([sx, my, mx, my, mx, iy, sx, iy, sx, my]),
          _rect([mx, wy, ix, wy, ix, iy, mx, iy, mx, wy]),
          _rect([wx, my, ix, my, ix, wy, wx, wy, wx, my])
        ]

    def __get_mask_rect_side(self, x, y):
        """Internal function. Returns a side where the cursor is on or None"""
        if self.mask_rect is None:
            return None

        p = (self.canvas.canvasx(x), self.canvas.canvasy(y))
        b = self.canvas.coords(self.mask_rect)

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

    def __draw_mask_rect(self):
        """Internal function. Draws a transparent mask part"""
        # Clean up
        if not self.mask_rect is None:
            self.canvas.delete(self.mask_rect)
            self.mask_rect = None

        # Draw rect
        sx = self.offset[0]
        sy = self.offset[1]
        mx = sx + self.mask[0]
        my = sy + self.mask[1]
        wx = sx + self.mask[2]
        wy = sy + self.mask[3]
        self.mask_rect = self.canvas.create_rectangle(
          mx, my,
          wx, wy,
          outline = self.mask_color,
          width = 2
        )

