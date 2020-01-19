#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     UI classes and functions
#
# Author:      kol
#
# Created:     04.07.2019
# Last change: 17.07.2020
# Copyright:   (c) kol 2019-2020
# Licence:     MIT
#-------------------------------------------------------------------------------
import os
from PIL import Image, ImageTk
from pathlib import Path
import random
import numpy as np
from imutils.perspective import four_point_transform
from collections import namedtuple
import sys
import tkinter as tk
from tkinter import ttk, font

from .utils import img_to_imgtk, resize3, is_on_w, board_spacing
from .binder import NBinder

UI_DIR = 'ui'    # directory containing ImgButton images
PADX = 5
PADY = 5
CV_WIDTH = 1
CV_HEIGTH = 0

# A label with additional tag
class NLabel(tk.Label):
    """Label with additional tag"""
    def __init__(self, master, tag=None, *args, **kwargs):
        tk.Label.__init__(self, master, *args, **kwargs)
        self.master, self.tag = master, tag

# A button which can load UI image and keep a reference to image provided
class NButton(tk.Button):
    """Button with image"""
    def __init__(self, master, *args, **kwargs):
        self.__uimage = kwargs.pop("uimage", None)
        self.__image = kwargs.get("image")
        if self.__uimage is not None and self.__image is None:
            self.__image = ImgButton.get_ui_image(self.__uimage)
            kwargs["image"] = self.__image
            if 'compound' not in kwargs:
                kwargs['compound'] = 'left'

        tk.Button.__init__(self, master, *args, **kwargs)


# ImageButton
class ImgButton(tk.Label):
    """Button with image face.
        This class represents a button with image and (currently) no text. A button has a
        tag which distinguish it from others and define what image is used as a face.
        The button can be toggled between states (normal/pressed).
        Upon clicking, a '<Click>' custom event is raised. A callback bound to the event
        could set event.cancel to True to prevent button from become pressed.
        Event parameters:
            event   TkInter event
            tag     Button tag
            state   New state (True if pressed, False otherwise).
        A button can be set to automatically create or close of a dialog window according to
        its state by providing a dlg_class parameter. In this case, <Dialog-Open> and
        <Dialog-Close> events are raised upon dialog showing up and closing.
        Event parameters:
            tag     Button tag
            dlg     Dialog been showing up or closed down
            ok      True if dialog is been closed after OK button press
    """
    # ImageButton event support classes
    class ImgButtonClickEvent:
        def __init__(self, event, tag, state):
            self.event, self.tag, self.state, self.cancel = event, tag, state, False

    ImgButtonDialogEvent = namedtuple('ImgButtonDialogEvent', ['tag', 'dlg', 'ok'])

    def __init__(self, *args, **kwargs):
        """Creates new ImgButton.

        Parameters:
            master      Tk windows/frame
            tag         Button tag. Files names "<tag>_down.png" and "<tag>_up.png" must exist in UI_DIR.
            state       Initial state (true/false)
            disabled    True/False
            tooltip     A tooltip text
            command     A callback function. See NBinder.bind()
            dlg_class   A dialog class descendant of GrDialog. If provided, it is assumed
                        that button is to be used to show/hide this dialog.
                        Note that if a callback is provided, dialog is created after it respecting results of the call.
                        A dialog should trigger <Close> event to allow button unpress upon close.
        """
        # Init
        self.__dlg = None
        self.__binder = NBinder()
        self.__DS_MAP = ['normal', 'disabled']

        # Parameters
        self.__tag = kwargs.pop('tag', None)
        if self.__tag is None:
            raise Exception('tag not provided')
        if 'callback' in kwargs:
            raise Exception("Using of 'callback' is deprecated, use 'command' or bind to <Click> event")
        self.__state = kwargs.pop('state', False)
        self.__disabled = kwargs.pop('disabled', False)
        tooltip = kwargs.pop('tooltip', None)
        self.__dlg_class = kwargs.pop('dlg_class', None)
        cmd = kwargs.pop('command', None)

        # Parent init
        tk.Label.__init__(self, *args, **kwargs)

        if not tooltip is None:
            self.__tooltip = createToolTip(self, tooltip)
        if cmd is not None:
            self.__binder.register(self, '<Click>', cmd)

        # Load button images
        self.__images = [ImgButton.get_ui_image(self.__tag + '_up.png'),
                         ImgButton.get_ui_image(self.__tag + '_down.png')]

        # Update configuration
        w = self.__images[0].width() + 4
        h = self.__images[0].height() + 4
        self.configure(borderwidth = 1, relief = "groove", width = w, height = h)
        self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

        self.__binder.bind(self, "<Button-1>", self.__mouse_click)

    @property
    def tag(self):
        """Button tag"""
        return self.__tag

    @property
    def state(self):
        """Button state"""
        return self.__state

    @state.setter
    def state(self, new_state):
        """Button state. Callbacks are not called when state is changed, use press/release/toggle"""
        if new_state != self.__state:
            self.__state = new_state
            self.configure(image = self.__images[new_state], state = self.__DS_MAP[self.__disabled])

    @property
    def disabled(self):
        """Button disabled state"""
        return self.__disabled

    @disabled.setter
    def disabled(self, ds):
        """Button disabled state"""
        if ds != self.__disabled:
            self.__disabled = ds
            self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

    @property
    def dialog(self):
        return self.__dlg

    def press(self):
        """Press a button """
        if not self.__state: self.toggle()

    def release(self):
        """Unpress a button """
        if self.__state: self.toggle()

    def toggle(self, event = None):
        """Toggle button state. Calls a callback and handle results of the call"""
        # Update state
        cur_state = self.__state
        new_state = not self.__state
        self.configure(image = self.__images[new_state], state = self.__DS_MAP[self.__disabled])
        if new_state: self._root().update()
        self.__state = new_state

        # Handle event
        e = self.ImgButtonClickEvent(event, self.__tag, new_state)
        self.__binder.trigger(self, '<Click>', e)
        if e.cancel:
            # Cancelling a state change
            self.__state = cur_state
            if new_state:
                # Unpress after small delay
                self.after(100, lambda: self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled]))
            else:
                self.configure(image = self.__images[self.__state], state = self.__DS_MAP[self.__disabled])

        # Handle a dialog communication
        if not self.__dlg_class is None:
            if self.__dlg is not None:
                self.__binder.unbind(self.__dlg, "<Close>")
                self.__binder.trigger(self, '<Dialog-Close>',
                                      self.ImgButtonDialogEvent(self.__tag, self.__dlg, False))
                dlg = self.__dlg
                self.__dlg = None
                dlg.close()

            if self.__state:
                self.__dlg = self.__dlg_class(master = self.master)
                self.__binder.register(self.__dlg, "<Close>", self.__dialog_close_callback)
                self.__binder.trigger(self, '<Dialog-Open>',
                                      self.ImgButtonDialogEvent(self.__tag, self.__dlg, False))

    def __mouse_click(self, event):
        """Mouse click callback"""
        if not self.__disabled: self.toggle()

    def __dialog_close_callback(self, event):
        """Dialog close callback"""
        if self.__dlg is not None:
            self.__binder.unbind(self.__dlg, "<Close>")
            self.__binder.trigger(self, '<Dialog-Close>',
                                  self.ImgButtonDialogEvent(self.__tag, self.__dlg, event.ok))
            self.__dlg = None
        self.state = False


    @staticmethod
    def get_ui_image(name):
        """Static method to get an image from UI directory"""
        ui_path = Path.cwd().joinpath(UI_DIR, name)
        return ImageTk.PhotoImage(Image.open(str(ui_path)))


# Button group
class ImgButtonGroup:
    """Button group management class.

        This class allows to define a button group can include several ImgButton tags
        and manage them as a whole (for example, disabling or unpressing them at once).
        Button groups can be dependent (where only one button could be down at one time)
        or independent, which are handled by the caller.
    """
    # Button group types
    BG_INDEPENDENT = "independent"
    BG_DEPENDENT = "dependent"


    # Enclosed group class
    class __Group:
        def __init__(self, parent, group, gtype, tags):
            self.parent, self.group, self.gtype, self.tags = parent, group, gtype, tags

        @property
        def buttons(self):
            return {b.tag: b for b in self.parent.get_buttons(self.group)}

        def __getitem__(self, key):
            return self.buttons[key]

        @property
        def state(self):
            return False

        @state.setter
        def state(self, s):
            self.parent.set_state(self.group, s)

        @property
        def disabled(self):
            return False

        @disabled.setter
        def disabled(self, d):
            self.parent.set_disabled(self.group, d)

        def release(self, exclude = None):
            self.parent.release(self.group, exclude)

    def __init__(self, master, **kwargs):
        """Create an instance.

        Parameters:
            master  A frame or ImagePanel containing ImgButtons
        """
        self.__master = master
        self.__binder = NBinder()
        self.__groups = dict()

    @property
    def buttons_list(self):
        """All image buttons displayed on master panel"""
        return self.get_buttons()

    @property
    def buttons(self):
        """Image button dictionary with tag as a key and button as value"""
        return {b.tag: b for b in self.get_buttons()}

    @property
    def groups(self):
        """Groups list"""
        return self.__groups

    def __getitem__(self, key):
        """Iterator for groups"""
        return self.__groups[key]

    @property
    def dependent_groups(self):
        """List of groups of type BG_DEPENDENT"""
        return self.get_groups(self.BG_DEPENDENT)

    def get_buttons(self, group = None, exclude = None):
        """List of image buttons displayed on master panel and, optionally, belonging to a group"""
        if group is not None and not group in self.groups:
            raise ValueError("Group '" + group + "' is not defined")

        frame = self.__master.headerPanel if (isinstance(self.__master, ImagePanel)) else self.__master
        return [c for c in frame.winfo_children() \
            if isinstance(c, ImgButton) and
                (group is None or c.tag in self.__groups[group].tags) and \
                (exclude is None or not c.tag in exclude) ]

    def get_groups(self, gtype = None):
        """Returns a list of groups, optionally, with specified type"""
        return [g for g in self.__groups if (gtype is None or self.__groups[g].gtype == gtype)]

    def add_group(self, group, tags, gtype = BG_INDEPENDENT):
        """Creates new button group"""
        if group in self.__groups:
            raise ValueError("Group '" + group + "' already defined")

        # Check group dependencies
        if gtype == self.BG_DEPENDENT:
            for g in self.dependent_groups:
                for t in self.__groups[g]:
                    if t in tags:
                        raise ValueError("Tag '" + t + "' was already included into dependent group '" + g + "'")

        # Register for ImgButton click event
        frame = self.__master.headerPanel if (isinstance(self.__master, ImagePanel)) else self.__master
        for c in frame.winfo_children():
            if isinstance(c, ImgButton) and c.tag in tags:
                self.__binder.register(c, "<Click>", self.__click_callback)

        # Store info
        self.__groups[group] = ImgButtonGroup.__Group(self, group, gtype, tags)

    def set_state(self, group, state, exclude = None):
        """Changes state of all buttons in a group"""
        g = self.get_buttons(group, exclude)
        for b in g: b.state = state

    def set_disabled(self, group, disabled, exclude = None):
        """Changes mode of all buttons in a group"""
        g = self.get_buttons(group, exclude)
        for b in g: b.disabled = disabled

    def release(self, group, exclude = None):
        """Releases all buttons in a group"""
        g = self.get_buttons(group, exclude)
        for b in g: b.release()

    def __click_callback(self, event):
        """Button click callback event"""
        if not event.state:
            # Unpressing, nothing to do
            return

        # Find dependent groups a clicked button belongs to
        grp = [g for g in self.dependent_groups if event.tag in self.__groups[g].tags]

        # Only one button in dependent group can be down at once
        for g in grp:
            for b in self.get_buttons(g):
                if b.tag != event.tag: b.release()

# Tooltip
class ToolTip:
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
    """Creates a tooltip with given text for given widget. Returns ToolTip instance"""
    toolTip = ToolTip(widget)
    widget.bind('<Enter>', lambda f: toolTip.showtip(text, coord))
    widget.bind('<Leave>', lambda f: toolTip.hidetip())
    return toolTip

# Remove a tooltip
def removeToolTip(toolTip):
    """Removes a tooltip"""
    toolTip.hidetip()
    toolTip.widget.unbind('<Enter>')
    toolTip.widget.unbind('<Leave>')

class StatusPanel(tk.Frame):
    """Status panel class"""

    def __init__(self, master, *args, **kwargs):
        """Creates StatusPanel instance. A StatusPanel is a frame with additional methods.

           Parameters:
               master           master frame
               max_width        maximum text width
               status           initial status
               callback         function to be called upon mouse click
        """
        self.__max_width = kwargs.pop('max_width', 0)
        self.__callback = kwargs.pop('callback', None)
        status = kwargs.pop('status', '')

        tk.Frame.__init__(self, master, *args, **kwargs)

        self.__var = tk.StringVar()
        self.__var.set(status)

        self.__label = tk.Label(self, textvariable = self.__var, anchor = tk.W)
        self.__label.pack(side = tk.LEFT, fill = tk.X, expand = True, anchor = tk.W)

        self.__binder = None
        if not self.__callback is None:
            self.__binder = NBinder()
            self.__binder.bind(self.__label, '<Button-1>', self.__callback)

    @property
    def status(self):
        """Status text"""
        return self.__var.get()

    @status.setter
    def status(self, text):
        """Status text"""
        self.set(text)

    @property
    def max_width(self):
        """Max status panel width in pixels"""
        return self.__max_width

    @max_width.setter
    def max_width(self, v):
        """Max status panel width in pixels"""
        self.__max_width = v

    def __get_maxw(self):
        """Internal function"""
        if self.__max_width == 0:
            return self.winfo_width()
        else:
            return min(self.__max_width, self.winfo_width())

    def set(self, text):
        """Set status as text. If text is larger than current panel size, it's been
        truncated from the end, ... added"""
        if self.__var.get() == text: return

        f = font.Font(font = self.__label['font'])
        chw = f.measure('W')

        maxw = self.__get_maxw()
        curw = f.measure(text)
        maxw -= chw*3
        if curw > maxw:
            strip_len = int((curw - maxw) / chw) + 3
            text = text[:-strip_len] + '...'

        self.__var.set(text)

    def set_file(self, text, file):
        """Set text + file name as status. If file name is too long, it's been
        shrinken by eliminating some path parts in the middle"""
        f = font.Font(font = self.__label['font'])
        chw = f.measure('W')

        maxw = self.__get_maxw() - chw*3
        if f.measure(text + file) > maxw:
            # Exclude file path parts to fit in starting from 3 entry
            parts = list(Path(file).parts)
            for n in range(len(parts)-3):
                parts.pop(len(parts)-2)
                t = parts[0] + '\\'.join(parts[1:])
                if f.measure(text + t) < maxw: break
            file = parts[0] + '\\'.join(parts[1:-2])
            file += '\\...\\' + parts[-1]

        self.__var.set(text + " " + file)

# Image panel class
class ImagePanel(tk.Frame):
    """A base class for displaying images"""

    # Resize event support class
    ResizeEvent = namedtuple('ResizeEvent', ['panel', 'old_scale', 'new_scale'])

    def __init__(self, master, **kwargs):
        """Creates ImagePanel instance.

           Parameters:
               master         master frame

           Keyworded parameters:
               caption          Panel caption
               image            OpenCv or PhotoImage. If none provided, an empty frame is created (default is None)
               frame_callback   Callback for panel mouse click (default is None)
               mode             Either 'clip' (default) or 'fit'.
                                If 'clip', the image is statically scaled to max_size and clipped to parent frame.
                                Scale can be changed by 'scale' property.
                                If 'fit, the image is scaled dynamically (respecting min_size and max_size). Parent frame
                                should have been aligned with pack(fill="both", expand = True). No scaling/panning is allowed.
               max_size         Maximum image size. If image is larger, it will be resized down to this size (default 0).
               min_size         Minimum image size when image is dynamically resized
               resize_callback  A function to be called after resizing. Takes event of ResizeEvent type.
               scrollbars       Boolean or tuple of booleans. If True both x and y scrollbars attached to canvas.
                                If tuple provided, it specify where scrollbars are attached (horiz, vert)
        """
        # will be assign after frame init
        self.__image = None
        self.__imgtk = None
        self.__src_image = None
        self.__scale = [1.0, 1.0]
        self.__offset = [0, 0]
        self.__image_shape = []

        # Panel parameters
        img = kwargs.pop('image', None)
        self.__caption = kwargs.pop('caption', None)
        frame_callback = kwargs.pop('frame_callback', None)
        f_sb = kwargs.pop('scrollbars', (False, False))
        if not type(f_sb) is tuple: f_sb = (f_sb, f_sb)
        self.__mode = kwargs.pop('mode',"clip")
        self.__max_size = kwargs.pop('max_size',  0)
        self.__min_size = kwargs.pop('min_size', 0)
        resize_callback = kwargs.pop('resize_callback', None)
        if self.__mode != "clip" and self.__mode != "fit":
            raise ValueError("Invalid mode ", self.__mode)
        if 'btn_params' in kwargs or 'btn_align' in kwargs:
            raise Exception("Using of 'btn_params' and 'btn_align' is deprecated, use add_button()")

        # Init
        tk.Frame.__init__(self, master, None, **kwargs)
        self.__src_image = img
        self.__set_image(img)

        # Panel to hold everything
        self.internalPanel = tk.Frame(self)
        self.internalPanel.pack(fill = tk.BOTH, expand = True)

        # Header panel and label
        self.headerPanel = None
        if self.__caption is not None:
            self.headerPanel = tk.Frame(self.internalPanel)
            self.headerPanel.pack(side = tk.TOP, fill = tk.X, expand = True)

        if self.__caption is not None and self.__caption != '':
            self.__header = tk.Label(self.headerPanel, text = self.__caption)
            self.__header.pack(side = tk.LEFT, fill = tk.X, expand = True)

        # Canvas
        canvasPanel = tk.Frame(self.internalPanel)
        canvasPanel.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        self.canvas = tk.Canvas(canvasPanel,
              width = self.max_canvas_size[0],
              height = self.max_canvas_size[1])
        self.canvas.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Image on canvas
        self.__image_id = None
        if not self.__imgtk is None:
            self.__image_id = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.__imgtk)

        # Scrollbars
        if f_sb[0]:
            sbb = tk.Scrollbar(self.internalPanel, orient=tk.HORIZONTAL)
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

        if not resize_callback is None:
            self.__binder.register(self, '<Resize>', resize_callback)

        # Resize handler
        if self.__mode == "fit":
           self.canvas.bind("<Configure>", self.__on_configure)

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
        return self.__imgtk

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
            old_scale = self.__scale
            self.__resize(size = self.__max_size, scale = scale)
            self.__update_image()
            new_scale = self.__scale
            self.__binder.trigger(self, "<Resize>", self.ResizeEvent(self, old_scale, new_scale))

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
    def max_canvas_size(self):
        """Maximum canvas size"""
        if not self.__image is None:
            return (self.__image.shape[CV_WIDTH], self.__image.shape[CV_HEIGTH])
        elif self.max_size > 0:
            return (self.max_size, self.max_size)
        else:
            return DEF_IMG_SIZE

    @property
    def image_shape(self):
        """Actual image shape"""
        return self.__image_shape

    @property
    def scaled_shape(self):
        """Image shape as it is displayed on canvas"""
        #return self.__image_shape if max(self.__scale) >= 1.0 else self.__image.shape
        return self.__image.shape

    @property
    def binder(self):
        """An event binder"""
        return self.__binder

    @property
    def buttons(self):
        """All image buttons residing on a headPanel"""
        buttons = {}
        for w in self.headerPanel.winfo_children():
            if isinstance(w, ImgButton):
                buttons[w.tag] = w
        return buttons

    def set_image(self, img):
        """Changes image

        Parameters:
            img             New image, either OpenCv or PhotoImage
        """
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
            self.__image_shape = [0, 0]
            self.__offset = [0, 0]
            self.__imgtk = None
        else:
            self.__image = image
            self.__image_shape = image.shape
            self.__resize()
            self.__imgtk = img_to_imgtk(self.__image)
            if hasattr(self, 'canvas'):
                self.canvas.config(
                    width=self.max_canvas_size[0],
                    height=self.max_canvas_size[1])

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
                          f_upsize = True,
                          f_center = True,
                          pad_color = (r, g, b))
            self.__imgtk = img_to_imgtk(self.__image)
            #print('{} -> {} x {} + {}'.format(orig_shape, self.__image.shape, self.__scale, self.__offset))

    def __update_image(self):
        """Internal function to update image"""
        if self.__imgtk is None and self.__image_id is not None:
            self.canvas.delete(self.__image_id)
        elif not self.__imgtk is None and self.__image_id is None:
            self.__image_id = self.canvas.create_image(0, 0, anchor = tk.NW, image = self.__imgtk)
        else:
            self.canvas.itemconfig(self.__image_id, image = self.__imgtk)

    def __on_configure(self, event):
        """ Event handler for resize events"""
        min_sz = min(event.width, event.height)
        max_sz = max(event.width, event.height)

        if self.__mode == "fit" and max_sz < self.__max_size and min_sz > self.__min_size:
            old_scale = self.__scale
            self.__resize(size = max_sz)
            self.__update_image()
            new_scale = self.__scale
            self.__binder.trigger(self, "<Resize>", self.ResizeEvent(self, old_scale, new_scale))

def addImagePanel(master, **kwargs):
    """Creates a panel with caption and buttons. Softly deprecated, provided for backward compatibility.
    See ImagePanel.init for arguments.

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

    # Constants
    MODE_AREA = 'area'
    MODE_GRID = 'grid'
    MODE_SPLIT = 'split'
    SIDE_LEFT = 0
    SIDE_TOP = 1
    SIDE_RIGHT = 2
    SIDE_BOTTOM = 3


    def __init__(self, panel, **kwargs):
        """Creates a mask object.

        Parameters:
            panel         ImagePanel reference
            mode          Either 'area' to use to set board area mask or 'grid' to set board grid. Default is 'area'
            size          Board size for 'grid' mode (default is 19)
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
        if self.__mode not in [self.MODE_AREA, self.MODE_GRID, self.MODE_SPLIT]:
           raise ValueError('Invalid mode', self.__mode)
        self.__size = kwargs.pop('size', 19)
        self.__mask = kwargs.pop('mask', None)
        self.__allow_change = kwargs.pop('allow_change', True)
        f_show = kwargs.pop('show_mask', False)
        self.__callback = kwargs.pop('mask_callback', None)

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
        if self.__mask is None:
            self.default_mask()
        if f_show: self.show()

        # Set handlers
        self.__bindings = NBinder()
        self.__bindings.register(self.__panel, "<Resize>", self.__on_panel_resize)
        if self.__allow_change:
            self.__bindings.bind(self.__panel.canvas, "<Motion>", self.motion_callback, add = "+")
            self.__bindings.bind(self.__panel.canvas, '<B1-Motion>', self.drag_callback, add = "+")
            self.__bindings.bind(self.__panel.canvas, '<B1-ButtonRelease>', self.end_drag_callback, add = "+")

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
        """Mask as it is displayed on canvas"""
        return np.array(self.__mask).astype('int').tolist() if self.__mask is not None else None

    @mask.setter
    def mask(self, m):
        """Mask as it is displayed on canvas"""
        self.hide()
        self.__mask = list(m) if m is not None else None

    @property
    def scaled_mask(self):
        """Mask scaled to actual image size"""
        if self.__mask is None:
           return None
        m = self.__mask.copy()
        m[0] = int(np.ceil(m[0] / self.__panel.scale[0]))
        m[1] = int(np.ceil(m[1] / self.__panel.scale[1]))
        m[2] = int(np.ceil(m[2] / self.__panel.scale[0]))
        m[3] = int(np.ceil(m[3] / self.__panel.scale[1]))
        return m

    @scaled_mask.setter
    def scaled_mask(self, mask):
        """Mask scaled to actual image size"""
        if mask is None:
           self.__mask = None
        else:
           # GrBoard() uses [[x1,y1],[x2,y2]] format, flattening required
           # mask is stored as double to prevent loss due to roudning
           m = np.array(mask).flatten().tolist()
           if len(m) < 4:
            raise ValueError('Insufficient length ', str(len(m)))
           m[0] = m[0] * self.__panel.scale[0]
           m[1] = m[1] * self.__panel.scale[1]
           m[2] = m[2] * self.__panel.scale[0]
           m[3] = m[3] * self.__panel.scale[1]
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
        """ImageMask mode ('area', 'grid', 'split')"""
        return self.__mode

    @mode.setter
    def mode(self, m):
        """ImageMask mode ('area', 'grid', 'split')"""
        m_ = m.lower()
        if m_ not in [self.MODE_AREA, self.MODE_GRID, self.MODE_SPLIT]:
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

    @property
    def is_shown(self):
        """ True if mask is currently shown"""
        return self.__mask_area is not None or self.__mask_rect is not None

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
            if self.__drag_side == self.SIDE_LEFT:
                self.__mask[0] = max(p[0], 0)
                if self.__mode == self.MODE_SPLIT:
                    self.__mask[2] = self.mask[0]
            elif self.__drag_side == self.SIDE_TOP:
                self.__mask[1] = max(p[1], 0)
            elif self.__drag_side == self.SIDE_RIGHT:
                self.__mask[2] = min(p[0], self.__panel.scaled_shape[1])
            elif self.__drag_side == self.SIDE_BOTTOM:
                self.__mask[3] = min(p[1], self.__panel.scaled_shape[0])

            # Reposition mask rect
            m = self.mask
            self.canvas.coords(self.__mask_rect,
                 m[0] + self.__panel.offset[0] + self.mask_width,
                 m[1] + self.__panel.offset[1] + self.mask_width,
                 m[2] + self.__panel.offset[0],
                 m[3] + self.__panel.offset[1])

            # Draw shading or grid
            if self.__mode == self.MODE_AREA:
                self.__draw_mask_shading()
            elif self.__mode == self.MODE_GRID:
                self.__draw_mask_grid()

    def end_drag_callback(self, event):
        """Callback for mouse button release event"""
        if not self.__drag_side is None and not self.__callback is None:
           self.__callback(self)
        self.__drag_side = None

    def show(self):
        """Draw a mask or grid on canvas"""
        if self.__mask is None: self.default_mask()

        if self.__mode == self.MODE_AREA:
            self.__draw_mask_shading()
            self.__draw_mask_rect()
        elif self.__mode == self.MODE_GRID:
            self.__draw_mask_grid()
        elif self.__mode == self.MODE_SPLIT:
            self.__draw_mask_split()

    def hide(self):
        """Hide a previously shown mask or grid"""
        if not self.__mask_area is None:
            for m in self.__mask_area:
                self.canvas.delete(m)
            self.__mask_area = None
        if not self.__mask_rect is None:
            self.canvas.delete(self.__mask_rect)
            self.__mask_rect = None

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
        dx = self.mask_width // 2
        dy = self.mask_width // 2
        self.__mask = [
                dx,
                dy,
                self.__panel.scaled_shape[CV_WIDTH] - 2*dx,
                self.__panel.scaled_shape[CV_HEIGTH] - 2*dy]
        if self.__mode == self.MODE_SPLIT:
            self.__mask[2] = self.__mask[2] // 2

    def __get_mask_rect_side(self, x, y):
        """Internal function. Returns a side where the cursor is on or None"""
        if self.__mask_rect is None:
            return None

        p = (self.canvas.canvasx(x), self.canvas.canvasy(y))
        b = self.canvas.coords(self.__mask_rect)

        side = None
        if is_on_w((b[0], b[1]), (b[0], b[3]), p, 2):
            side = self.SIDE_LEFT
        elif is_on_w((b[0], b[1]), (b[2], b[1]), p, 2):
            side = self.SIDE_TOP
        elif is_on_w((b[2], b[1]), (b[2], b[3]), p, 2):
            side = self.SIDE_RIGHT if self.__mode != self.MODE_SPLIT else self.SIDE_LEFT
        elif is_on_w((b[0], b[3]), (b[2], b[3]), p, 2):
            side = self.SIDE_BOTTOM

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
        """Internal function. Draws a mask as an area"""
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
        """Internal function. Draws a mask as a grid"""

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

    def __draw_mask_split(self):
        """Internal function. Draws a mask as a splitter"""
        # Clean up
        if not self.__mask_rect is None:
            self.canvas.delete(self.__mask_rect)
            self.__mask_rect = None

        # Draw rect
        m = self.mask
        sx = self.__panel.offset[0]
        sy = self.__panel.offset[1]
        mx = sx + m[2]
        my = sy + m[1] + self.mask_width
        wx = sx + m[2]
        wy = sy + m[3]

        self.__mask_rect = self.canvas.create_rectangle(
          mx, my,
          wx, wy,
          outline = self.mask_color,
          width = self.mask_width
        )

    def __on_panel_resize(self, e):
        """Callback for panel resize (internal function)"""
        #print("{} -> {}".format(old_scale, new_scale))
        if self.__mask is None:
            return

        m = self.__mask.copy()
        m[0] = m[0] / e.old_scale[0] * e.new_scale[0]
        m[1] = m[1] / e.old_scale[1] * e.new_scale[1]
        m[2] = m[2] / e.old_scale[0] * e.new_scale[0]
        m[3] = m[3] / e.old_scale[1] * e.new_scale[1]
        self.__mask = m
        if self.is_shown: self.show()


# Image transformer
class ImageTransform:
    """4-points image transformation helper class.
    After calling start(), it displays a tooltip and waits till a left button will
    be clicked 4 times on a image panel. After 4th click, it runs 4-point transformation
    with clicked coordinates as transformation rectangle. Results are either
    displayied in the image panel or provided to the caller in callback.
    """

    def __init__(self, panel, **kwargs):
        """Create ImageTransorm instance

            Parameters:
                panel       An ImagePanel object with Image displayed on
                inplace     If True (default), transformed image will be shown in image panel
                connect     If True, transformation points will be connected by a line.
                            Line parameters can be set with connect_xxx properties.
                            Default is False.
                keep        If True, transformation points and lines will remain displayed
                            after transformation finished.
                            To hide them, explicit call to hide() or cancel() is required.
                            Default is False. If inplace is True, this parameter is ignored.
                allow_change  If True, transformatio points can be moved with the mouse.
                            Default is True. If keep is False, this parameter is ignored.
                            Each move will trigger new transformation
                show_coord  If True, click coordinates will be displayed
                callback    A callback function to be called upon transform completed or cancelled.
                            Function signature:
                                callback(transformer, image) where:
                                         tranformer  ImageTranform object
                                         image       Transformed image or None
                                                      if transformation was cancelled.
        """
        self.__panel = panel

        self.__id = random.randrange(0, 100)
        self.__transform_state = False
        self.__transform_rect = None
        self.__transform_help = None
        self.__transform_scale = None
        self.__transform_offset = None
        self.__drag_point = None

        self.__src_image = None
        self.__bindings = NBinder()

        # Public attributes
        self.inplace = kwargs.pop('inplace', True)
        self.callback = kwargs.pop('callback', None)
        self.connect = kwargs.pop('connect', False)
        self.keep = kwargs.pop('keep', False)
        self.allow_change = kwargs.pop('allow_change', True)
        self.show_coord =  kwargs.pop('show_coord', False)

        self.dot_color = "red"      # Color and fill of a transformation point
        self.coord_color = "red"    # Color of coordinates text
        self.connect_color = "red"  # Color of line between transformation points
        self.connect_dash = (6, 4)  # Connection line dash pattern

    @property
    def panel(self):
        return self.__panel

    @property
    def canvas(self):
        return self.__panel.canvas

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
    def transform_image(self):
        """An image after transformation"""
        return self.__get_transform_image()

    @property
    def transform_rect(self):
        """Transformation rectangle (TL, TR, BL, BR) as it is displayed on screen"""
        if self.__transform_rect is None:
           return None
        else:
           t = [t[:2].view(int).tolist() for t in self.__transform_rect]
           return t

    @transform_rect.setter
    def transform_rect(self, tr):
        """Transformation rectangle (TL, TR, BL, BR) as it is displayed on screen"""
        if tr is None:
            self.__transform_rect = None
        else:
            self.__transform_rect = np.array(tr)

    @property
    def scaled_rect(self):
        """Transformation rectangle (TL, TR, BL, BR) scaled to actual image size"""
        if self.transform_rect is None:
            return None
        else:
            t = []
            scale = self.__transform_scale if self.__transform_scale is not None \
                                           else self.__panel.scale
            offset = self.__transform_offset if self.__transform_offset is not None \
                                           else self.__panel.offset

            for r in self.transform_rect:
                t.append([
                    int(np.ceil(r[0] / scale[0])) - offset[0],
                    int(np.ceil(r[1] / scale[1])) - offset[1] ])
            return t

    @scaled_rect.setter
    def scaled_rect(self, tr):
        """Transformation rectangle (TL, TR, BL, BR) scaled to actual image size"""
        if tr is None:
            self.__transform_rect = None
        else:
            t = []
            for r in tr:
                t.append([
                    int(np.ceil(r[0] * self.__panel.scale[0])) + self.__panel.offset[0],
                    int(np.ceil(r[1] * self.__panel.scale[1])) + self.__panel.offset[1]])
            self.__transform_rect = np.array(t)
            self.__transform_scale = None
            self.__transform_offset = None

    @property
    def bounding_rect(self):
        """Bounding transformation rectangle (scaled)"""
        t = self.scaled_rect
        if t is None:
            return None
        else:
            min_x = min([x[0] for x in t])
            max_x = max([x[0] for x in t])
            min_y = min([x[1] for x in t])
            max_y = max([x[1] for x in t])
            return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]

    def start(self):
        """Initiates a transform operation"""
        if self.__transform_state:
           self.cancel()

        # Clean up
        self.__clean_up()

        # Register bindings, display help message
        self.__bindings.bind(self.canvas, "<Button-1>", self.__mouse_callback)
        self.__bindings.bind(self.panel.winfo_toplevel(), "<Escape>", self.__key_callback)

        bb_help = self.canvas.bbox(tk.ALL)
        cx = int((bb_help[0] + bb_help[2])/2)
        cy = int((bb_help[1] + bb_help[3])/2)
        self.__transform_help = createToolTip(self.canvas,
            'Click on 4 image corners or press ESC to cancel', (cx, cy))

        # Initiate user actions
        self.__transform_state = True
        self.__src_image = self.__panel.src_image
        self.canvas.after(100, self.__check_transform_state)
        return True

    def cancel(self):
        """Cancel transformation which was already started"""
        self.__clean_up()
        if not self.callback is None:
            self.callback(self, None)

    def show(self):
        """Show transformation area (works only if keep is True)"""
        if self.__transform_rect is not None:
            for n, r in enumerate(self.__transform_rect):
                self.__show_point(n, r[0], r[1])

    def hide(self):
        """Hide transformation area (works only if keep is True)"""
        self.__drag_point = None
        if self.__transform_rect is not None:
            self.canvas.delete(self.transform_tag)
            for r in self.__transform_rect:
                for n in range(2, len(r)): r[n] = 0

    def reset(self):
        """Display source image on a panel. Does nothing if inplace is False"""
        if self.__src_image is not None and self.inplace:
            if self.__transform_state: self.cancel()
            self.panel.image = self.__src_image

    @property
    def transform_tag(self):
        """Returns an unique identifier of the transformation object"""
        return 'transform_' + str(self.__id).zfill(3)

    def __check_transform_state(self):
        """Internal function - timer callback"""
        if self.__transform_state:
            # Still running, repeat check
            self.canvas.after(100, self.__check_transform_state)
        else:
            if not self.__transform_rect is None:
                # Save current scale and offset since they will change with image change
                self.__transform_scale = self.panel.scale
                self.__transform_offset = self.panel.offset

                # Do 4-points transform
                img = self.__get_transform_image()
                if self.inplace:
                    self.panel.set_image(img)
                if self.callback is not None:
                    self.callback(self, img)

            # Clean up if keeping drawn items was not requested
            # Otherwise keep them on
            # If changed them is allowed, register mouse dragging events
            self.__clean_up(clear_points = not self.keep or self.inplace)
            if self.keep and not self.inplace and self.allow_change:
                self.__drag_point = None
                self.__bindings.bind(self.canvas, "<B1-Motion>", self.__drag_callback)
                self.__bindings.bind(self.canvas, '<B1-ButtonRelease>', self.__end_drag_callback)

    def __mouse_callback(self, event):
        """Internal function - mouse click callback"""
        if not self.__transform_state: return

        if self.__transform_rect is None:
            # Draw 1st point
            self.__transform_rect = np.zeros((4, 5), dtype = np.int32)
            self.__show_point(0, event.x, event.y)
        else:
            # Find empty position (up to 4)
            for n, r in enumerate(self.__transform_rect):
                if r[2] == 0:
                    # Empty slot, occupy and draw items
                    self.__show_point(n, event.x, event.y)
                    self.__transform_state = (n < len(self.__transform_rect)-1)
                    break

    def __key_callback(self, event):
        """Internal function - ESC key press callback"""
        self.cancel()

    def __drag_callback(self, event):
        """Internal function - mouse dragging callback"""
        if self.__drag_point is None:
            # Determine point been dragged
            # If a point was hidden, it has 0 item IDs - ignore them
            for n, r in enumerate(self.__transform_rect):
                if abs(event.x - r[0]) < 8 and abs(event.y - r[1]) < 8 and r[2] > 0:
                    self.__drag_point = n

        if self.__drag_point is not None:
            p = (int(self.canvas.canvasx(event.x) - self.panel.offset[0]),
                 int(self.canvas.canvasy(event.y) - self.panel.offset[1]))

            # Move items drawn on canvas
            # Note: panel shape is in OpenCV indexing scheme (y first)
            self.__move_point(self.__drag_point,
                min(max(p[0], 0), self.panel.scaled_shape[1]),
                min(max(p[1], 0), self.panel.scaled_shape[0]))

    def __end_drag_callback(self, event):
        """Internal function - mouse release callback"""
        if self.__drag_point is not None:
            self.__drag_point = None
            if not self.inplace and self.callback is not None:
                img = self.__get_transform_image()
                self.callback(self, img)

    def __show_point(self, index, x, y):
        """Internal function - show a new transformation point and given coordinates"""
        self.__transform_rect[index, 0] = x
        self.__transform_rect[index, 1] = y

        # Transformation point
        circle_id = self.canvas.create_oval(
            x - 4, y - 4,
            x + 4, y + 4,
            fill=self.dot_color,
            outline=self.dot_color,
            tags=self.transform_tag)
        self.__transform_rect[index, 2] = circle_id

        # Connection line
        if self.connect:
            if index > 0:
                # Connect to previous point
                prev_n = index - 1
                x2 = self.__transform_rect[prev_n, 0]
                y2 = self.__transform_rect[prev_n, 1]
                line_id = self.canvas.create_line(
                    x, y, x2, y2,
                    fill=self.connect_color,
                    dash=self.connect_dash,
                    tags=self.transform_tag)
                self.__transform_rect[index, 3] = line_id

            if index == len(self.__transform_rect)-1:
                # Last point, connect to 1st point
                x2 = self.__transform_rect[0, 0]
                y2 = self.__transform_rect[0, 1]
                line_id = self.canvas.create_line(
                    x, y, x2, y2,
                    fill=self.connect_color,
                    dash=self.connect_dash,
                    tags=self.transform_tag)
                self.__transform_rect[0, 3] = line_id

        # Coordnates
        if self.show_coord:
            text_id = self.canvas.create_text(
                x, y + 10,
                text = "({},{})".format(x, y),
                fill=self.coord_color,
                tags=self.transform_tag)
            self.__transform_rect[index, 4] = text_id

    def __move_point(self, index, nx, ny):
        """Internal function - move a new transformation point to new coordinates"""
        x, y, circle_id, line_id, text_id = self.__transform_rect[index]

        # Transformation point
        if circle_id > 0:
            self.canvas.coords(circle_id, nx - 4, ny - 4, nx + 4, ny + 4)

        # Connection line
        if line_id > 0:
            prev_n = index - 1 if index > 0 else len(self.__transform_rect)-1
            x2 = self.__transform_rect[prev_n, 0]
            y2 = self.__transform_rect[prev_n, 1]
            self.canvas.coords(line_id, nx, ny, x2, y2)

            next_n = index + 1 if index < len(self.__transform_rect)-1 else 0
            line_id2 = self.__transform_rect[next_n, 3]
            x2 = self.__transform_rect[next_n, 0]
            y2 = self.__transform_rect[next_n, 1]
            self.canvas.coords(line_id2, x2, y2, nx, ny)

        # Coordnates
        if text_id > 0:
            self.canvas.coords(text_id, nx, ny + 10)
            self.canvas.itemconfig(text_id, text="({},{})".format(nx, ny))

        self.__transform_rect[index, 0] = nx
        self.__transform_rect[index, 1] = ny

    def __get_transform_image(self):
        """Internal function - retrieve transformed image"""
        if self.__transform_rect is None or len(self.__transform_rect) < 4:
            raise ValueError('Transformation rectangle not defined')

        t = np.array([t[:2] for t in self.__transform_rect])
        return four_point_transform(self.image, t)

    def __clean_up(self, clear_points = True):
        """Internal function - clear state"""
        self.__transform_state = False
        self.__drag_point = None

        if clear_points and self.__transform_rect is not None:
            # Remove selection points
            self.canvas.delete(self.transform_tag)
            self.__transform_rect = None

        # Remove tooltip
        if not self.__transform_help is None:
            removeToolTip(self.__transform_help)
            self.__transform_help = None

        # Cancel bindings
        self.__bindings.unbind_all()

# Dialog window
class GrDialog(tk.Toplevel):
    """A base class for dialog and supplementary windows.
    The class creates a window positioned next to root,
    either as modal or non modal dialog. The window will have an internal
    frame where controls should be placed and a button bar at the bottom.
    By default, the button bar will have either Close or OK and Cancel buttons
    depending on ok_cancel internal flag.
    Upon closing, the window raises a <Close> custom event with 'dlg' (self)
    and 'ok' (whether OK button was pressed) parameters.
    Subclasses could override the following methods:
        get_minsize()   Returns (min_x, min_y) tuple specifying minimum window size
                        Note that TkInter fits a window to children's bounding box,
                        so actual windows size could be large
        get_title()     Returns a title string
        get_position()  Returns a (x, y) tuple where to display a window in
                        screen coordinates. By default uses get_offset() results to
                        position the window relative to root
        get_offset()    Return a (off_x, off_y) tuple for offsetting to root window
        init_params()   Called before TkInter initialization to retrieve additional
                        parameters from args or kwargs
        init_state()    Called after TkInter initialization to init initernal vars
        init_frame()    Should be overridden to add controls to internal frame
        init_buttons()  Should be overridden to add controls to button frame.
                        Default method adds Close or OK+Cancel buttons depending on
                        ok_cancel internal flag value (False by default)
        grab_focus()    Should make window modal or just take focus:
                            If self.focus_set() is called, focus is assigned to window
                            If self.grab_set() is called, window becomes modal
                        Default method calls focus_set() and also self.resizable(False, False)
        update_controls()   Called when focus is assigned to window after loss,
                            should be overriden to update internal state
        close()         Called upon window destruction. By default, triggers
                        <Close> custom event passing reference to self (as dlg)
                        and OK pressing flag (as ok). Default method should be called
                        if overriden
        """

    def __init__(self, *args, **kwargs):
        # Initialization
        self.binder = NBinder()
        self.ok_cancel = False
        self.ok = False

        self.init_params(args, kwargs)
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.init_state()

        # Window mechanics
        self.transient(self.get_root())
        self.attributes("-toolwindow", True)

        m = self.get_minsize()
        p = self.get_position()
        self.title(self.get_title())
        self.minsize(m[0], m[1])
        self.geometry("%+d%+d" % (p[0], p[1]))

        # Bindings
        self.bind("<Escape>", self.escape_callback)
        self.bind("<FocusIn>", self.focus_in_callback)
        self.protocol("WM_DELETE_WINDOW", self.on_closing_callback)

        # Payload
        self.internalFrame = tk.Frame(self)
        self.internalFrame.pack(side = tk.TOP, fill = tk.BOTH, expand = True)

        self.buttonFrame = tk.Frame(self, bd = 1, relief = tk.RAISED)
        self.buttonFrame.pack(side = tk.BOTTOM, fill = tk.X)

        self.init_frame(self.internalFrame)
        self.init_buttons(self.buttonFrame)

        # Focus
        self.grab_focus()

    def get_root(self):
        """Returns reference to root application window of tk.Tk class"""
        m = self.master
        while m is not None:
            if isinstance(m, tk.Tk): return m
            m = m.master
        return None

    @property
    def root(self):
        """A root application window of tk.Tk class"""
        return self.get_root()

    def get_minsize(self):
        """Override to define minimum window size"""
        return (300, 300)

    def get_title(self):
        """Override to define window caption"""
        return "Dialog"

    def get_position(self):
        """Override to set predefined position. Alternative - to override get_offset()"""
        ofs = self.get_offset()
        return (self.get_root().winfo_x() + self.get_root().winfo_width() + ofs[0],
            self.get_root().winfo_y() + ofs[1])

    def get_offset(self):
        """Override to define offset from to parent window"""
        return (15, 40)

    def init_params(self, args, kwargs):
        """Override to get extra parameters from arguments"""
        pass

    def init_state(self):
        """Override to setup after widget initialization done"""
        pass

    def init_frame(self, internalFrame):
        """Override to add controls to internal frame"""
        pass

    def init_buttons(self, buttonFrame):
        """Override to add buttons to button frame. Inherited method should be called."""
        if not self.ok_cancel:
            tk.Button(self.buttonFrame, text = "Close",
                command = self.close_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)
        else:
            tk.Button(self.buttonFrame, text = "OK",
                command = self.ok_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)
            tk.Button(self.buttonFrame, text = "Cancel",
                command = self.close_click_callback).pack(side = tk.LEFT, padx = 5, pady = 5)

    def update_controls(self):
        """Override to update controls on getting focus"""
        pass

    def grab_focus(self):
        """Override to change how focus is set"""
        self.focus_set()
        self.resizable(False, False)

    def ok_click_callback(self):
        """OK button click callback"""
        self.ok = True
        self.close()

    def close_click_callback(self):
        """Close button click callback"""
        self.ok = False
        self.close()

    def escape_callback(self, event):
        """Escape key press callback"""
        self.close()

    def focus_in_callback(self, event):
        """Window got focus"""
        self.update_controls()

    def on_closing_callback(self):
        """Window been closed"""
        self.close()

    def close(self):
        """Graceful way to close the dialog.
        Override to add custom logic, inherited method should be called"""
        Event = namedtuple('Event', ['dlg', 'ok'])
        self.binder.trigger(self, "<Close>", Event(dlg = self, ok = self.ok))
        self.binder.unbind_all()
        self.destroy()

    def update_position(self):
        """Update window position"""
        m = self.get_minsize()
        p = self.get_position()
        self.minsize(m[0], m[1])
        self.geometry("%+d%+d" % (p[0], p[1]))

    def enable_controls(self, state, cb_def_state = "normal"):
        """Enables or disables controls hosted at internal frame"""
        def f(frame):
            for w in frame.winfo_children():
                if isinstance(w, tk.Frame):
                    f(w)
                elif isinstance(w, ttk.Combobox):
                    w.configure(state = cb_def_state if state else "disabled")
                elif isinstance(w, ImgButton):
                    w.disabled = not state
                elif isinstance(w, (ttk.Widget, tk.Widget)):
                    w.configure(state = tk.NORMAL if state else tk.DISABLED)
        f(self.internalFrame)

# A marker on an ImagePanel
class ImageMarker:
    """A marker on a panel"""

    def __init__(self, panel, **kwargs):
        """Create ImageMarker instance

            Parameters:
                panel       An ImagePanel object with Image displayed on
                stones      List of stones in GR format (list of lists [x,y,a,b,r])
                show_stones True to show stones immediatlly (True if stones provided)
                marker      Marker type (currently only 0 or "circle" supported)
                radius      Radius to draw circles (overrides radius in stone list)
                flash       Number of times to flash selection when a new stone is added
        """
        self.__panel = panel
        self.__stones = kwargs.pop('stones', [])
        show = kwargs.pop('show_stones', len(self.__stones) > 0)
        self.__type = kwargs.pop('marker', 0)
        self.__radius = kwargs.pop('radius', 0)
        self.__flash = kwargs.pop('flash', 0)
        self.last_stone = None

        self.__markers = []
        if show: self.show()

        self.__bindings = NBinder()
        self.__bindings.register(self.__panel, "<Resize>", self.__on_panel_resize)

        # Public properies
        self.line_color = { "_": "red", "B": "deep sky blue", "W": "purple2"}
        self.fill_color = { "_": "", "B": "deep sky blue", "W": "purple2"}
        self.line_width = { "_": 2, "B": 1, "W": 1}
        self.fill_stipple = { "_": "", "B": "gray50", "W": "gray50"}

    @property
    def panel(self):
        """ImagePanel"""
        return self.__panel

    @property
    def canvas(self):
        """Canvas"""
        return self.__panel.canvas

    @property
    def stones(self):
        return self.__stones.copy()

    def add_stone(self, stone, bw = None, f_show = True, f_replace = False):
        """Add stone"""
        if f_replace and len(self.__stones) > 0: self.clear()
        st = stone.copy()
        if len(stone) <= GR_BW:
            if bw is None:
                raise Exception('Stone color is not specified')
            st.extend(bw)
        self.__stones.extend([stone])

        if f_show:
            if self.__flash == 0:
                self.__draw_marker(stone)
            else:
                self.__flash_marker(stone, self.__flash, True)

    def add_stones(self, stones, bw = None, f_show = True, f_replace = False):
        """Add stone from list"""
        if f_replace and len(self.__stones) > 0: self.clear()
        if stones is None or len(stones) == 0: return

        if len(stones[0]) > GR_BW:
            # Stones has bw property
            self.__stones.extend(list(stones))
        elif bw is not None:
            # Color provided provided
            self.__stones.extend([list(x).extend(bw) for x in stones])
        else:
            raise Exception('Stone color is not specified')

        if f_show:
            self.__draw_markers()

    def del_stone(self, stone):
        """Remove a stone"""
        for i, s in enumerate(self.__stones):
            if s[0][GR_A] == stone[GR_A] and s[0][GR_B] == stone[GR_B]:
                self.__stones.pop(i)
        if self.is_shown: self.__draw_markers()

    def clear(self):
        """Remove all stones"""
        self.hide()
        self.__stones = []

    def show(self):
        """Show markers"""
        self.__draw_markers()

    def hide(self):
        """Hide previously shown markers"""
        if not self.__markers is None:
            for m in self.__markers:
                self.canvas.delete(m)
            self.__markers = []

    @property
    def is_shown(self):
        """ True if markers are currently shown"""
        return self.__markers is not None and len(self.__markers) > 0

    def __draw_marker(self, stone):
        """Internal function. Draws a marker for a stone"""
        def circle(x, y, r, bw):
            return self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline = self.line_color[bw],
                fill = self.fill_color[bw],
                width = self.line_width[bw]
            )

        def circle_poly(x, y, r, bw):
            nr_points = 32
            angles = np.linspace(0, 2*np.pi, nr_points+1)[0:-1]
            points = []
            for angle in angles:
                points.append(x + r * np.cos(angle))
                points.append(y + r * np.sin(angle))

            return self.canvas.create_polygon(
                  *points,
                  outline = self.line_color[bw],
                  width = self.line_width[bw],
                  fill = self.fill_color[bw],
                  stipple = self.fill_stipple[bw])

        p = self.panel.image2frame((stone[GR_X], stone[GR_Y]))
        r = stone[GR_R] if self.__radius == 0 else self.__radius
        r = int(r * self.__panel.scale[0])
        bw = stone[GR_BW] if len(stone) > GR_BW and stone[GR_BW] is not None else '_'

        if self.fill_stipple[bw] == "":
            m = circle(p[0], p[1], r, bw)
        else:
            m = circle_poly(p[0], p[1], r, bw)
        self.__markers.extend([m])

    def __draw_markers(self):
        """Internal function. Draws all markers"""
        # Clean up
        if not self.__markers is None:
            for m in self.__markers :
                self.canvas.delete(m)
        self.__markers = []

        # Draw
        for stone in self.__stones:
            self.__draw_marker(stone)

    def __on_panel_resize(self, e):
        """Callback for panel resize (internal function)"""
        self.__draw_markers()

    def __flash_marker(self, stone, cnt, on_off):
        if on_off:
            # Marker is ON
            self.__draw_marker(stone)
            if cnt > 0:
                self.canvas.after(100, lambda: self.__flash_marker(stone, cnt, False))
        elif len(self.__markers) > 0:
            # Marker is OFF
            m = self.__markers.pop(-1)
            self.canvas.delete(m)
            self.canvas.after(100, lambda: self.__flash_marker(stone, cnt-1, True))

