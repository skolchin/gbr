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


# GUI class
class GbrGUI:
      def __init__(self, root):
          self.root = root

          # Defaults params
          self.grParams = grdef.DEF_GR_PARAMS.copy()
          self.grRes = None
          self.showBlack = True
          self.showWhite = True

          # Default board image and generated image
          self.origImg = gr.generate_board()
          self.origImgName = None
          self.origImgTk = grutils.img_to_imgtk(self.origImg)
          self.genImg = self.origImg
          self.genImgTk = grutils.img_to_imgtk(self.genImg)

          # Images panel (original + generated + analysis)
          self.imgFrame = tk.Frame(self.root)
          self.imgFrame.grid(row = 0, column = 0, padx = PADX, pady = PADY)

          # Image panel: original image (label + photo)
          self.origImgLabel = tk.Label(self.imgFrame, text = "Original")
          self.origImgLabel.grid(row = 0, column = 0, sticky = "nswe")

          self.origImgPanel = tk.Label(self.imgFrame, image = self.origImgTk)
          self.origImgPanel.bind('<Button-1>', self.orig_img_mouse_callback)
          self.origImgPanel.grid(row = 1, column = 0, sticky = "nswe", padx = PADX)

          # Image panel: generated image header
          self.genLblFrame = tk.Frame(self.imgFrame)
          self.genLblFrame.grid(row = 0, column = 1, sticky = "nswe")
          tk.Grid.columnconfigure(self.genLblFrame, 0, weight = 1)

          # Image panel: generated image header: label + buttons
          self.genImgLabel = tk.Label(self.genLblFrame, text = "Generated")
          self.genImgLabel.grid(row = 0, column = 0, padx = PADX, pady = PADY, sticky = "nswe")

          self.blackImgTk = [ImageTk.PhotoImage(Image.open('black_up.png')),
                             ImageTk.PhotoImage(Image.open('black_down.png'))]
          self.showBlackBtn = tk.Label(self.genLblFrame, image = self.blackImgTk[1],
                                                         borderwidth = 1,
                                                         relief = "groove",
                                                         width = 30)
          self.showBlackBtn.bind("<Button-1>", self.show_black_callback)
          self.showBlackBtn.grid(row = 0, column = 1, padx = PADX, pady = PADY, sticky = "nswe")

          self.whiteImgTk = [ImageTk.PhotoImage(Image.open('white_up.png')),
                             ImageTk.PhotoImage(Image.open('white_down.png'))]
          self.showWhiteBtn = tk.Label(self.genLblFrame, image = self.whiteImgTk[1],
                                                         borderwidth = 1,
                                                         relief = "groove",
                                                         width = 30)
          self.showWhiteBtn.bind("<Button-1>", self.show_white_callback)
          self.showWhiteBtn.grid(row = 0, column = 2, padx = PADX, pady = PADY, sticky = "nswe")

          # Image panel: generated image panel
          self.genImgPanel = tk.Label(self.imgFrame, image = self.origImgTk)
          self.genImgPanel.bind('<Button-1>', self.gen_img_mouse_callback)
          self.genImgPanel.grid(row = 1, column = 1, sticky = "nswe", padx = PADX)

          # Image panel: analysis images
          self.dbgImgLabel = tk.Label(self.imgFrame, text = "Analysis")
          self.dbgImgLabel.grid(row = 0, column = 2, sticky = "nwe")

          self.dbgFrameRoot = tk.Frame(self.imgFrame)
          self.dbgFrameRoot.grid(row = 1, column = 2, padx = PADX, sticky = "nswe")

          self.dbgFrameCanvas = tk.Canvas(self.dbgFrameRoot)
          self.dbgFrameCanvas.pack(side = tk.LEFT)
          self.dbgFrameScroll= tk.Scrollbar(self.dbgFrameRoot, command=self.dbgFrameCanvas.yview)
          self.dbgFrameScroll.pack(side=tk.LEFT, fill='y')

          self.dbgFrameCanvas.configure(yscrollcommand = self.dbgFrameScroll.set)
          self.dbgFrameCanvas.bind('<Configure>', self.on_scroll_configure)

          self.dbgFrame = tk.Frame(self.dbgFrameCanvas)
          self.dbgFrameCanvas.create_window((0,0), window=self.dbgFrame, anchor='nw')

          # Info frame
          self.infoFrame = tk.Frame(self.root, width = self.origImg.shape[grdef.CV_WIDTH]*2, height = 300)
          self.infoFrame.grid(row = 1, column = 0, padx = PADX, pady = PADY, sticky = "nswe")

          # Info frame: buttons
          self.buttonFrame = tk.Frame(self.infoFrame, bd = 1, relief = tk.RAISED,
                                                 width = self.origImg.shape[grdef.CV_WIDTH]*2+PADX*2, height = 50)
          self.buttonFrame.grid(row = 0, column = 0, sticky = "nswe")
          self.buttonFrame.grid_propagate(0)

          self.loadImgBtn = tk.Button(self.buttonFrame, text = "Load image",
                                                        command = self.load_img_callback)
          self.loadImgBtn.grid(row = 0, column = 0, padx = PADX, pady = PADY)

          self.saveParamBtn = tk.Button(self.buttonFrame, text = "Save params",
                                                          command = self.save_json_callback)
          self.saveParamBtn.grid(row = 0, column = 1, padx = PADX, pady = PADY)

          self.saveBrdBtn = tk.Button(self.buttonFrame, text = "Save board",
                                                        command = self.save_jgf_callback)
          self.saveBrdBtn.grid(row = 0, column = 2, padx = PADX, pady = PADY)

          self.applyBtn = tk.Button(self.buttonFrame, text = "Apply",
                                                      command = self.apply_callback)
          self.applyBtn.grid(row = 0, column = 3, padx = PADX, pady = PADY)

          self.applyDefBtn = tk.Button(self.buttonFrame, text = "Defaults",
                                                         command = self.apply_def_callback)
          self.applyDefBtn.grid(row = 0, column = 4, padx = PADX, pady = PADY)

          # Info frame: stones info
          self.boardInfo = tk.StringVar()
          self.boardInfo.set("No stones found")
          self.boardInfoPanel = tk.Label(self.buttonFrame, textvariable = self.boardInfo)
          self.boardInfoPanel.grid(row = 0, column = 5, sticky = "nwse", padx = PADX)

          # Info frame: switches
          self.switchFrame = tk.Frame(self.infoFrame, bd = 1, relief = tk.RAISED)
          self.switchFrame.grid(row = 1, column = 0, sticky = "nswe")
          self.tkVars = self.add_switches(self.switchFrame, self.grParams)

          # Status bar
          self.statusFrame = tk.Frame(self.root, width = 200, bd = 1, relief = tk.SUNKEN)
          self.statusFrame.grid(row = 2, column = 0, sticky = "nswe")

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

          if self.origImgName is None:
            return

          x = x - int((w - self.origImg.shape[grdef.CV_WIDTH]) / 2)
          y = y - int((h - self.origImg.shape[grdef.CV_HEIGTH]) / 2)
          print('{}, {}'.format(x, y))

          f = "Black"
          p = gr.find_coord(x, y, self.grRes[grdef.GR_STONES_B])
          if (p[0] == -1):
            f = "White"
            p = gr.find_coord(x, y, self.grRes[grdef.GR_STONES_W])
          if (p[0] >= 0):
            ct = "{f} {a}{b} at ({x},{y})".format(
               f = f,
               a = grutils.stone_pos(p, grdef.GR_A),
               b = grutils.stone_pos(p, grdef.GR_B),
               x = round(p[grdef.GR_X],0),
               y = round(p[grdef.GR_Y],0))
            print(ct)
            self.stoneInfo.set(ct)

      # Callback for mouse events on original image
      def orig_img_mouse_callback(self, event):
          self.load_img_callback()

      # Callback for mouse event on debug image
      def dbg_img_mouse_callback(self, event):
        w = event.widget
        k = w.tag
        grutils.show(k, self.grRes[k])

      # Load image button callback
      def load_img_callback(self):
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if (fn != ""):
           # Load the image
           self.origImg = cv2.imread(fn)
           self.origImgTk = grutils.img_to_imgtk(self.origImg)
           self.origImgPanel.configure(image = self.origImgTk)
           self.origImgName = fn

           # Load JSON with image recog parameters
           ftitle = ""
           fnj = Path(self.origImgName).with_suffix('.json')
           if fnj.is_file():
              p = json.load(open(fnj))
              ftitle = " (with params)"
              for key in self.grParams.keys():
                  if p.get(key) is not None:
                     self.grParams[key] = p[key]
                     if key in self.tkVars:
                        self.tkVars[key].set(self.grParams[key])

           # Process image
           self.showBlack = True
           self.showWhite = True
           self.update_board(reprocess = True)

           # Update status
           self.stoneInfo.set("File loaded{ft}: {fn}".format(ft = ftitle, fn = str(self.origImgName)))

      # Save params button callback
      def save_json_callback(self):
        # Save json with current parsing parameters
        if self.origImgName is None:
           # Nothing to do!
           return

        fn = Path(self.origImgName).with_suffix('.json')
        with open(fn, "w", encoding="utf-8", newline='\r\n') as f:
             json.dump(self.grParams, f, indent=4, sort_keys=True, ensure_ascii=False)

        self.stoneInfo.set("Params saved to: " + str(fn))

      # Save stones button callback
      def save_jgf_callback(self):
        # Save json with current parsing parameters
        if self.origImgName is None:
           # Nothing to do!
           return

        jgf = grutils.gres_to_jgf(self.grRes)
        fn = Path(self.origImgName).with_suffix('.jgf')
        with open(fn, "w", encoding="utf-8", newline='\r\n') as f:
             json.dump(jgf, f, indent=4, sort_keys=True, ensure_ascii=False)

        self.stoneInfo.set("Stones saved to: " + str(fn))

      # Apply button callback
      def apply_callback(self):
        for key in self.tkVars.keys():
            self.grParams[key] = self.tkVars[key].get()
        self.update_board(reprocess = True)

      # Apply defaults button callback
      def apply_def_callback(self):
        self.grParams = grdef.DEF_GR_PARAMS.copy()
        for key in self.tkVars.keys():
            self.tkVars[key].set(self.grParams[key])
        self.update_board(reprocess = True)

      # Callback for canvas configuration
      def on_scroll_configure(self, event):
        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        self.dbgFrameCanvas.configure(scrollregion=self.dbgFrameCanvas.bbox('all'))

      # Callback for "Show black stones"
      def show_black_callback(self, event):
        if self.origImgName is None:
           return

        self.showBlack = not self.showBlack
        self.showBlackBtn.configure(image = self.blackImgTk[int(self.showBlack)])
        self.update_board(reprocess= False)

      # Callback for "Show white stones"
      def show_white_callback(self, event):
        if self.origImgName is None:
           return

        self.showWhite = not self.showWhite
        self.showWhiteBtn.configure(image = self.whiteImgTk[int(self.showWhite)])
        self.update_board(reprocess= False)

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
      def add_debug_info(self, root, shape, res):
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
      def update_board(self, reprocess = True):
        # Process original image
        if self.grRes is None or reprocess:
           self.grRes = gr.process_img(self.origImg, self.grParams)

        # Generate board using analysis results
        r = self.grRes.copy()
        if not self.showBlack:
           del r[grdef.GR_STONES_B]
        if not self.showWhite:
           del r[grdef.GR_STONES_W]

        self.genImg = gr.generate_board(shape = self.origImg.shape, res = r)
        self.genImgTk = grutils.img_to_imgtk(self.genImg)
        self.genImgPanel.configure(image = self.genImgTk)

        black_stones = self.grRes[grdef.GR_STONES_B]
        white_stones = self.grRes[grdef.GR_STONES_W]
        board_size = self.grRes[grdef.GR_BOARD_SIZE]
        self.boardInfo.set("Board size: {}, black stones: {}, white stones: {}".format(
                                  board_size, black_stones.shape[0], white_stones.shape[0]))

        # Update debug info
        self.add_debug_info(self.dbgFrame, self.origImg.shape, self.grRes)


# Main function
def main():
    # Construct interface
    window = tk.Tk()
    window.title("Go board")
    gui = GbrGUI(window)

    # Main loop
    window.mainloop()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
