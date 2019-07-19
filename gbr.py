#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Main and interfaces functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

import grdef
import gr
import grutils

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import string as ss
import json
from pathlib import Path

# Constants
PADX = 5
PADY = 5

# Image frame with additional tag - for debug info
class NLabel(tk.Label):
      def __init__(self, master, tag=None, *args, **kwargs):
          tk.Label.__init__(self, master, *args, **kwargs)
          self.master, self.tag = master, tag

# Global variables
origImg = None                # originally loaded image
origImgName = None            # name of original image file
window = None                 # root window
origImgPanel = None           # panel to display original image
genImgPanel = None            # generated image panel
boardInfo = None              # board info panel
stoneInfo = None              # stone information panel
tkVars = None                 # list of trackbard linked to board recog params
dbgFrame = None               # Frame to hold list of debug images
canvas = None                 # Canvas around debug image
grParams = None               # list of board recog parameters
grRes = None                  # Board recognition results
showBlack = True             # Show/hide black stones
showWhite = True             # Show/hide white stones

# Main function
def main():
     # Callback functions
    # Callback for mouse events on generated board image
    def gen_img_mouse_callback(event):
        global grRes
        global origImg

        # Convert from widget coordinates to image coordinates
        w = event.widget.winfo_width()
        h = event.widget.winfo_height()
        x = event.x
        y = event.y

        if grRes is None or origImg is None:
           return

        x = x - int((w - origImg.shape[grdef.CV_WIDTH]) / 2)
        y = y - int((h - origImg.shape[grdef.CV_HEIGTH]) / 2)
        print('{}, {}'.format(x, y))

        f = "Black"
        p = gr.find_coord(x, y, grRes[grdef.GR_STONES_B])
        if (p[0] == -1):
           f = "White"
           p = gr.find_coord(x, y, grRes[grdef.GR_STONES_W])
        if (p[0] >= 0):
           ct = "{f} {a}{b} at ({x},{y})".format(
              f = f,
              a = ss.ascii_uppercase[p[2]-1],
              b = p[3],
              x = round(p[0],0),
              y = round(p[1],0))
           print(ct)
           stoneInfo.set(ct)

    # Callback for mouse events on original image
    def orig_img_mouse_callback(event):
        load_img_callback()

    # Callback for mouse event on debug image
    def dbg_img_mouse_callback(event):
        global grRes

        w = event.widget
        k = w.tag
        grutils.show(k, grRes[k])


    # Load image button callback
    def load_img_callback():
        global origImg
        global origImgName

        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if (fn != ""):
           # Load the image
           origImg = cv2.imread(fn)
           imgtk = grutils.img_to_imgtk(origImg)
           origImgPanel.configure(image = imgtk)
           origImgPanel.image = imgtk
           origImgName = fn

           # Load JSON with image recog parameters
           ftitle = ""
           fnj = Path(origImgName).with_suffix('.json')
           if fnj.is_file():
              p = json.load(open(fnj))
              ftitle = " (with params)"
              for key in grParams.keys():
                  if p.get(key) is not None:
                     grParams[key] = p[key]
                     if tkVars.get(key) is not None:
                        tkVars[key].set(grParams[key])

           # Process image
           showBlack = True
           showWhite = True
           update_board()

           # Update status
           stoneInfo.set("File loaded{ft}: {fn}".format(ft = ftitle, fn = str(origImgName)))

    # Save params button callback
    def save_json_callback():
        global origImgName

        # Save json with current parsing parameters
        if origImgName is None or origImgName == "":
           # Nothing to do!
           return

        fn = Path(origImgName).with_suffix('.json')
        with open(fn, "w", encoding="utf-8", newline='\r\n') as f:
             json.dump(grParams, f, indent=4, sort_keys=True, ensure_ascii=False)

        stoneInfo.set("Params saved to: " + str(fn))

    # Apply button callback
    def apply_callback():
        for key in tkVars.keys():
            grParams[key] = tkVars[key].get()
        update_board()

    # Apply defaults button callback
    def apply_def_callback():
        global grParams

        grParams = grdef.DEF_GR_PARAMS.copy()
        for key in tkVars.keys():
            tkVars[key].set(grParams[key])

        update_board()

    # Callback for canvas configuration
    def on_configure(event):
        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        canvas.configure(scrollregion=canvas.bbox('all'))

    # Callback for "Show black stones"
    def show_black_callback():
        global showBlack
        global origImg

        if origImg is None:
           return

        showBlack = not showBlack
        update_board(reprocess= False)

    # Callback for "Show white stones"
    def show_white_callback():
        global showWhite
        global origImg

        if origImg is None:
           return

        showWhite = not showWhite
        update_board(reprocess= False)

    # Add Scale widgets with board recognition parameters
    def add_switches(root, params, nrow = 0):
        n = 1
        ncol = 0
        frame = None
        vars = dict()

        # Add a tabbed notebook
        nb = ttk.Notebook(root)
        nb.grid(row = nrow, column = 0, sticky = "nswe", padx = PADX, pady = PADY)

        # Get unique tabs
        tabs = set([e[2] for e in grdef.GR_PARAMS_PROP.values() if e[2]])

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
                if grdef.GR_PARAMS_PROP[key][2] == tab:
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
                    panel = tk.Scale(frame, from_ = grdef.GR_PARAMS_PROP[key][0],
                                            to = grdef.GR_PARAMS_PROP[key][1],
                                            orient = tk.HORIZONTAL,
                                            variable = v)
                    panel.grid(row = n, column = 1, padx = 2, pady = 2)
                    vars[key] = v

                    n = n + 1
        return vars


    # Add analysis results info
    def add_debug_info(root, shape, res):
        if res is None:
           return

        nrow = 0
        ncol = 0
        sx = int(shape[grdef.CV_WIDTH] / 2) - 5
        sy = sx

        # Remove all previously added controls
        for c in root.winfo_children():
            c.destroy()

        # Add analysis result images
        for key in res.keys():
            if key.find("IMG_") >= 0:
               frame = tk.Frame(root)
               frame.grid(row = nrow, column = ncol, padx = 2, pady = 2, sticky = "nswe")

               img = cv2.resize(res[key], (sx, sy))

               imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
               panel = NLabel(frame, image = imgtk, tag = key)
               panel.image = imgtk
               panel.grid(row = 0, column = 0)
               panel.bind('<Button-1>', dbg_img_mouse_callback)

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

        edges = res[grdef.GR_EDGES]
        spacing = res[grdef.GR_SPACING]
        hcross = res[grdef.GR_NUM_CROSS_H]
        vcross = res[grdef.GR_NUM_CROSS_W]
        size = res[grdef.GR_BOARD_SIZE]

        lbox.insert(tk.END, "Edges: ({},{}) : ({},{})".format(edges[0][0], edges[0][1], edges[1][0], edges[1][1]))
        lbox.insert(tk.END, "Net: {},{}".format(round(spacing[0],2), round(spacing[1],2)))
        lbox.insert(tk.END, "Cross: {},{}".format(hcross, vcross))
        lbox.insert(tk.END, "Size: {}".format(size))

        panel = tk.Label(frame, text = "TEXT_INFO")
        panel.grid(row = 1, column = 0, sticky = "nswe")

    # Update board
    def update_board(reprocess = True):
        global grRes
        global origImg

        # Process original image
        if grRes is None or reprocess:
           grRes = gr.process_img(origImg, grParams)

        # Generate board using analysis results
        r = grRes.copy()
        if not showBlack:
           del r[grdef.GR_STONES_B]
        if not showWhite:
           del r[grdef.GR_STONES_W]

        gen_img = gr.generate_board(shape = origImg.shape, res = r)
        imgtk = grutils.img_to_imgtk(gen_img)
        genImgPanel.configure(image = imgtk)
        genImgPanel.image = imgtk

        black_stones = grRes[grdef.GR_STONES_B]
        white_stones = grRes[grdef.GR_STONES_W]
        board_size = grRes[grdef.GR_BOARD_SIZE]
        boardInfo.set("Board size: {}, black stones: {}, white stones: {}".format(board_size, black_stones.shape[0], white_stones.shape[0]))

        # Update debug info
        add_debug_info(dbgFrame, origImg.shape, grRes)


    # ===
    # Main code
    # ===

    # Defaults params
    grParams = grdef.DEF_GR_PARAMS.copy()

    # Make default board image
    origImg = gr.generate_board()

    # Main window
    window = tk.Tk()
    window.title("Go board")

    # Image panel
    imgFrame = tk.Frame(window)
    imgFrame.grid(row = 0, column = 0, padx = PADX, pady = PADY)

    # Image panel: original image
    panel = tk.Label(imgFrame, text = "Original")
    panel.grid(row = 0, column = 0, sticky = "nswe")

    imgtk = grutils.img_to_imgtk(origImg)
    origImgPanel = tk.Label(imgFrame, image = imgtk)
    origImgPanel.image = imgtk
    origImgPanel.bind('<Button-1>', orig_img_mouse_callback)
    origImgPanel.grid(row = 1, column = 0, sticky = "nswe", padx = PADX)

    # Image panel: generated image
    panel = tk.Label(imgFrame, text = "Generated")
    panel.grid(row = 0, column = 1, sticky = "nswe")

    genImgPanel = tk.Label(imgFrame, image = imgtk)
    genImgPanel.image = imgtk
    genImgPanel.bind('<Button-1>', gen_img_mouse_callback)
    genImgPanel.grid(row = 1, column = 1, sticky = "nswe", padx = PADX)

    # Image panel: analysis images
    panel = tk.Label(imgFrame, text = "Analysis")
    panel.grid(row = 0, column = 2, sticky = "nwe")

    frame = tk.Frame(imgFrame)
    frame.grid(row = 1, column = 2, padx = PADX, sticky = "nswe")

    canvas = tk.Canvas(frame)
    canvas.pack(side = tk.LEFT)
    scrollbar = tk.Scrollbar(frame, command=canvas.yview)
    scrollbar.pack(side=tk.LEFT, fill='y')

    canvas.configure(yscrollcommand = scrollbar.set)
    canvas.bind('<Configure>', on_configure)

    dbgFrame = tk.Frame(canvas)
    canvas.create_window((0,0), window=dbgFrame, anchor='nw')

    # Info frame
    infoFrame = tk.Frame(window, width = origImg.shape[grdef.CV_WIDTH]*2, height = 300)
    infoFrame.grid(row = 1, column = 0, padx = PADX, pady = PADY, sticky = "nswe")
    #infoFrame.grid_propagate(0)

    # Info frame: buttons
    buttonFrame = tk.Frame(infoFrame, bd = 1, relief = tk.RAISED, width = origImg.shape[grdef.CV_WIDTH]*2+PADX*2, height = 50)
    buttonFrame.grid(row = 0, column = 0, sticky = "nswe")
    buttonFrame.grid_propagate(0)

    panel = tk.Button(buttonFrame, text = "Load image", command = load_img_callback)
    panel.grid(row = 0, column = 0, padx = PADX, pady = PADY)

    panel = tk.Button(buttonFrame, text = "Save params", command = save_json_callback)
    panel.grid(row = 0, column = 1, padx = PADX, pady = PADY)

    panel = tk.Button(buttonFrame, text = "Apply", command = apply_callback)
    panel.grid(row = 0, column = 2, padx = PADX, pady = PADY)

    panel = tk.Button(buttonFrame, text = "Defaults", command = apply_def_callback)
    panel.grid(row = 0, column = 3, padx = PADX, pady = PADY)

    panel = tk.Button(buttonFrame, text = "Black", command = show_black_callback)
    panel.grid(row = 0, column = 4, padx = PADX, pady = PADY)

    panel = tk.Button(buttonFrame, text = "White", command = show_white_callback)
    panel.grid(row = 0, column = 5, padx = PADX, pady = PADY)

    # Info frame: stones info
    boardInfo = tk.StringVar()
    boardInfo.set("No stones found")
    panel = tk.Label(buttonFrame, textvariable = boardInfo)
    panel.grid(row = 0, column = 6, sticky = "nwse", padx = PADX)

    # Info frame: switches
    switchFrame = tk.Frame(infoFrame, bd = 1, relief = tk.RAISED)
    switchFrame.grid(row = 1, column = 0, sticky = "nswe")
    tkVars = add_switches(switchFrame, grParams)

    # Status bar
    statusFrame = tk.Frame(window, width = 200, bd = 1, relief = tk.SUNKEN)
    statusFrame.grid(row = 2, column = 0, sticky = "nswe")

    stoneInfo = tk.StringVar()
    stoneInfo.set("")
    panel = tk.Label(statusFrame, textvariable = stoneInfo)
    panel.grid(row = 0, column = 0, sticky = tk.W, padx = 5, pady = 2)

    # Main loop
    window.mainloop()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
