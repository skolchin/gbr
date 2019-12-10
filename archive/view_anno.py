#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network dataset review
#
# Author:      kol
#
# Created:     03.08.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
import numpy as np
import cv2
import sys
from PIL import Image, ImageTk
from pathlib import Path
from gr.utils import resize2
from gr.board import GrBoard
from gr.ui_extra import *
import logging
from gr.dataset import GrDataset
from gr.grlog import GrLog

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog  as filedialog
    import ttk
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk

class ViewAnnoGui:
    def __init__(self, root, max_size = 550, allow_open = True):
        self.root = root
        self.zoom = [1.0, 1.0]
        self.allow_open = allow_open
        self.f_rect = True

        # Top frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.buttonFrame = tk.Frame(self.root, width = max_size + 10, height = 70, bd = 1, relief = tk.RAISED)
        self.buttonFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)
        self.configFrame = tk.Frame(self.root, bd = 1, relief = tk.RAISED)
        self.configFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)
        self.statusFrame = tk.Frame(self.root, width = max_size + 2*PADX, bd = 1, relief = tk.SUNKEN)
        self.statusFrame.pack(side = tk.BOTTOM, fill=tk.BOTH, padx = PADX, pady = PADY)

        # Image frame
        self.defBoardImg = GrBoard(board_shape = (max_size, max_size)).image
        self.boardImgName = None
        self.annoName = None
        self.srcName = None

        self.imgPanel = addImagePanel(self.imgFrame,
                caption = "Dataset image",
                btn_params = [["box", True, self.show_rec_callback, "Rectangle/circle"]],
                image = self.defBoardImg,
                frame_callback = self.open_img_callback,
                max_size = max_size)
        self.imgPanel.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        # Config
        self.dataset = GrDataset.getDataset()

        tk.Label(self.configFrame, text = "Train image size").grid(row = 0, column = 0)
        self.trainSize = tk.StringVar()
        self.trainSize.set(str(self.dataset.image_size['train']))
        self.trainSizeEntry = tk.Entry(self.configFrame, textvariable = self.trainSize)
        self.trainSizeEntry.grid(row = 0, column = 1, padx = PADX, pady = PADY)

        tk.Label(self.configFrame, text = "Test image size").grid(row = 1, column = 0)
        self.testSize = tk.StringVar()
        self.testSize.set(str(self.dataset.image_size['test']))
        self.testSizeEntry = tk.Entry(self.configFrame, textvariable = self.testSize)
        self.testSizeEntry.grid(row = 1, column = 1, padx = PADX, pady = PADY)

        # Buttons
        if self.allow_open:
            self.openBtn = tk.Button(self.buttonFrame, text = "Open",
                                                          command = self.open_btn_callback)
            self.openBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        self.updateBtn = tk.Button(self.buttonFrame, text = "Update",
                                                      command = self.update_callback)
        self.updateBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        self.updateAllBtn = tk.Button(self.buttonFrame, text = "Dataset update",
                                                      command = self.update_all_callback)
        self.updateAllBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        self.showLogBtn = tk.Button(self.buttonFrame, text = "Show log",
                                                      command = self.show_log_callback)
        self.showLogBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        # Img info
        self.imgInfo = tk.StringVar()
        self.imgInfo.set("")
        self.imgnfoPanel = tk.Label(self.buttonFrame, textvariable = self.imgInfo)
        self.imgnfoPanel.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        # Status panel
        self.statusInfo = addStatusPanel(self.statusFrame, max_size + 2*PADX)
        self.statusInfo.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)


    # Load annotation button callback
    def open_btn_callback(self):
        if not self.allow_open: return  # GUI used from other app

        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("Annotation files","*" + self.dataset.anno_ext),("All files","*.*")))
        if fn != "": self.load_annotation(fn)

    # Image click callback
    def open_img_callback(self, event):
        self.open_btn_callback()

    # Board update button callback
    def update_callback(self):
        if not self.annoName is None:
           self.update_img_size()
           self.update_anotation()

    # Dataset regeneration button callback
    def update_all_callback(self):
        self.update_img_size()

        GrLog.clear()
        try:
            self.dataset.generate_dataset()
            if GrLog.numErrors() > 0:
                self.statusInfo.set("Errors during processing, see the log")
            else:
                self.statusInfo.set("Dataset updated")
        except:
            logging.exception('Error')
            self.statusInfo.set("Errors during processing, see the log")
            return

    def show_rec_callback(self, event, tag, state):
        if self.annoName is None:
            return False
        else:
            self.f_rect = state
            self.load_annotation(self.annoName)
            return True

    def show_log_callback(self):
        GrLog.show(self.root)

    def load_annotation(self, file):
        """Load annotation from file"""

        GrLog.clear()
        try:
            # Load annotation file
            fn, src, shape, bboxes = self.dataset.load_annotation(file)

            # Load image
            img = cv2.imread(fn)
            if img is None:
                raise Exception('File not found {}'.format(fn))

            # Resize the image
            self.boardImgName = fn
            self.annoName = file
            self.srcName = src
            self.imgFrame.pack_propagate(False)

            self.imgPanel.image = img          # this adopts image to frame max_size
            img2 = self.imgPanel.image         # image to draw upon

            # Process objects
            for bb in bboxes:
                # Get coordinates
                p1 = self.imgPanel.image2frame((bb[0][0],bb[0][1]))
                p2 = self.imgPanel.image2frame((bb[1][0],bb[1][1]))
                cls = bb[2]

                # Draw a bounding box
                clr = (0,0,255)
                if cls == "black": clr = (255,0,0)

                if self.f_rect:
                    cv2.rectangle(img2, p1, p2, clr,1)
                else:
                    d = max(p2[0]-p1[0], p2[1]-p1[1])
                    x = int(p1[0] + d/2)
                    y = int(p1[1] + d/2)
                    cv2.circle(img2, (x,y), int(d/2), clr, 1)

            self.imgPanel.image = img2      # display image with drawing on the panel

            # Update status
            stage = self.dataset.get_stage(self.boardImgName)
            img_info = "Size: ({}, {}), stage: {}".format(img.shape[1], img.shape[0], stage)
            self.imgInfo.set(img_info)

            if GrLog.numErrors() > 0:
                self.statusInfo.set("Errors during processing, see the log")
            else:
                self.statusInfo.set_file('File loaded: ', self.annoName)
        except:
            logging.exception('Error')
            self.statusInfo.set("Errors during processing, see the log")

    def update_anotation(self):
        """Updates annotation for currently loaded file"""
        GrLog.clear()

        # Check whether JGF exists
        # If not - image goes to test dataset, stones should not be stored in annotation
        jgf_file = Path(self.dataset.src_path).joinpath(Path(self.boardImgName).name).with_suffix('.jgf')
        f_process = jgf_file.is_file()
        logging.info('Board file {} exists: {}'.format(jgf_file, f_process))

        try:
            # Load board from annotation file
            # This will find the image and load it to the board
            board = GrBoard()
            board.load_annotation(self.annoName, self.dataset, f_process = f_process)

            # Determine stage to store image to
            stage = self.dataset.get_stage(self.boardImgName)
            logging.info('Current stage {}'.format(stage))
            if stage is None:
               stage = 'test'
               if not board.results is None: stage = 'train'

            # Save annortation
            board.save_annotation(self.annoName, self.dataset, anno_only = False, stage = stage)

            # Update status
            stage = self.dataset.get_stage(self.boardImgName)
            img_info = "Size: ({}, {}), stage: {}".format(self.imgPanel.image.shape[1], \
                                                          self.imgPanel.image.shape[0], \
                                                          stage)
            self.imgInfo.set(img_info)

            if GrLog.numErrors() > 0:
                self.statusInfo.set("Errors during processing, see the log")
            else:
                self.statusInfo.set("Dataset updated")
        except:
            logging.exception('Error')
            self.statusInfo.set("Errors during processing, see the log")
            return


    def update_img_size(self):
        """Updates dataset image parameters"""
        try:
            n1 = int(self.trainSize.get())
            n2 = int(self.testSize.get())
            if n1 == 0 or n1 > 300:
                self.dataset.image_size['train'] = n1
            else:
                raise Exception("Invalid value")
            if n2 == 0 or n2 > 300:
                self.dataset.image_size['test'] = n2
            else:
                raise Exception("Invalid value")
            return True
        except:
            self.statusInfo.set("Image size should be integer equal to zero or greater than 300")
            return False


def main():
    window = tk.Tk()
    window.title("View annotaitons")

    log = GrLog.init()
    gui = ViewAnnoGui(window)

    window.mainloop()

if __name__ == '__main__':
    main()

