#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network dataset review
#
# Author:      skolchin
#
# Created:     03.08.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import cv2
import sys
from PIL import Image, ImageTk
from pathlib import Path
from gr.utils import img_to_imgtk, resize2
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

        # Set paths
        self.root_path = Path(__file__).parent.resolve()
        self.src_path = self.root_path.joinpath("img")
        self.ds_path = self.root_path.joinpath("gbr_ds")
        if not self.ds_path.exists(): self.ds_path.mkdir(parents = True)
        self.meta_path = self.ds_path.joinpath("data","Annotations")
        if not self.meta_path.exists(): self.meta_path.mkdir(parents = True)
        self.img_path = self.ds_path.joinpath("data","Images")
        if not self.img_path.exists(): self.img_path.mkdir(parents = True)
        self.sets_path = self.ds_path.joinpath("data","ImageSets")
        if not self.sets_path.exists(): self.sets_path.mkdir(parents = True)

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
        self.boardImg = self.defBoardImg
        self.boardImgTk = img_to_imgtk(self.boardImg)
        self.boardImgName = None
        self.annoName = None
        self.srcName = None

        _, self.imgPanel, _ = addImagePanel(self.imgFrame,"Dataset image",
                [["box", True, self.show_rec_callback, "Rectangle/circle"]],
                self.boardImgTk, self.open_img_callback)

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

        # Status frame
        self.statusInfo = addStatusPanel(self.statusFrame, self.defBoardImg.shape[1])
        self.statusInfo.grid(row = 0, column = 0, sticky = tk.W, padx = 5, pady = 2)


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
            img2, self.zoom = resize2 (img, np.max(self.defBoardImg.shape[:2]), f_upsize = False)

            # Process objects
            for bb in bboxes:
                x1 = bb[0][0]
                y1 = bb[0][1]
                x2 = bb[1][0]
                y2 = bb[1][1]
                cls = bb[2]

                # Draw a bounding box
                x1 = int(x1 * self.zoom[0])
                x2 = int(x2 * self.zoom[0])
                y1 = int(y1 * self.zoom[1])
                y2 = int(y2 * self.zoom[1])
                clr = (0,0,255)
                if cls == "black": clr = (255,0,0)

                if self.f_rect:
                    cv2.rectangle(img2,(x1,y1),(x2,y2),clr,1)
                else:
                    d = max(x2-x1, y2-y1)
                    x = int(x1 + d/2)
                    y = int(y1 + d/2)
                    cv2.circle(img2, (x,y), int(d/2), clr, 1)

            # Display the image
            self.boardImg = img2
            self.boardImgTk = img_to_imgtk(img2)
            self.boardImgName = fn
            self.annoName = file
            self.srcName = src
            self.imgFrame.pack_propagate(False)
            self.imgPanel.configure(image = self.boardImgTk)

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
        jgf_file = self.src_path.joinpath(Path(self.boardImgName).name).with_suffix('.jgf')
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
            img_info = "Size: ({}, {}), stage: {}".format(self.boardImg.shape[1], \
                                                          self.boardImg.shape[0], \
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

