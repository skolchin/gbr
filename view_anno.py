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
import xml.dom.minidom as minidom
from gr.utils import img_to_imgtk, resize2
from gr.board import GrBoard
from gr.ui_extra import *
import logging
from make_dataset import generate_dataset, default_image_size
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

        _, self.imgPanel, _ = addImagePanel(self.imgFrame,"Dataset image",
                [["box", True, self.show_rec_callback, "Rectangle/circle"]],
                self.boardImgTk, self.open_img_callback)

        # Config
        self.dsImgSize = default_image_size()

        tk.Label(self.configFrame, text = "Train image size").grid(row = 0, column = 0)
        self.trainSize = tk.StringVar()
        self.trainSize.set(str(self.dsImgSize['train']))
        self.trainSizeEntry = tk.Entry(self.configFrame, textvariable = self.trainSize)
        self.trainSizeEntry.grid(row = 0, column = 1, padx = PADX, pady = PADY)

        tk.Label(self.configFrame, text = "Test image size").grid(row = 1, column = 0)
        self.testSize = tk.StringVar()
        self.testSize.set(str(self.dsImgSize['test']))
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
        self.imgInfo .set("")
        self.imgnfoPanel = tk.Label(self.buttonFrame, textvariable = self.imgInfo)
        self.imgnfoPanel.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        # Status frame
        self.statusInfo = tk.StringVar()
        self.statusInfo.set("")
        self.stoneInfoPanel = tk.Label(self.statusFrame, textvariable = self.statusInfo)
        self.stoneInfoPanel.grid(row = 0, column = 0, sticky = tk.W, padx = 5, pady = 2)

        # Load datasets
        self.datasets = None
        self.load_datasets()
        pass

    # Load annotation button callback
    def open_btn_callback(self):
        if not self.allow_open: return  # GUI used from other app

        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("XML files","*.xml"),("All files","*.*")))
        if fn != "": self.load_annotation(fn)

    # Image click callback
    def open_img_callback(self, event):
        self.open_btn_callback()

    # Board update button callback
    def update_callback(self):
        if self.annoName is None:
            return

        # Init
        GrLog.clear()
        self.update_img_size()

        # Check whether JGF exists
        # If not - image goes to test dataset, stones should not be stored in annotation
        jgf_file = self.src_path.joinpath(Path(self.boardImgName).name).with_suffix('.jgf')
        f_process = jgf_file.is_file()
        logging.info('Board file {} exists: {}'.format(jgf_file, f_process))

        try:
            # Load board from annotation file
            # This will find the image and load it to the board
            board = GrBoard()
            board.load_annotation(self.annoName, path_override = str(self.src_path), f_process = f_process)

            # Get dataset
            ds = self.find_dataset(self.boardImgName)
            logging.info('Current dataset {}'.format(ds))
            if ds is None:
               ds = 'test'
               if not board.results is None: ds = 'train'

            # Resize image to dataset preferred size
            if self.dsImgSize[ds] > 0:
               logging.info('Resizing image to {}'.format(self.dsImgSize[ds]))
               board.resize_board(self.dsImgSize[ds])

            # Save image to dataset path
            png_file = self.img_path.joinpath(Path(board.image_file).with_suffix('.png').name)
            board.save_image(str(png_file))
            board.save_annotation(self.annoName)

            # Include file into proper dataset
            ds = self.find_dataset(self.boardImgName)
            if ds is None:
               ds = 'test'
               if not board.results is None: ds = 'train'
               self.datasets[ds].append(str(png_file.stem))
               self.save_datasets()
               logging.info('New dataset {}'.format(ds))
            elif ds == 'test' and not board.results is None:
               self.datasets[ds].remove(str(png_file.stem))
               ds = 'train'
               self.datasets[ds].append(str(png_file.stem))
               self.save_datasets()
               logging.info('New dataset {}'.format(ds))

            if GrLog.numErrors() > 0:
                self.stoneInfo.set("Errors during processing, see the log")
            else:
                self.statusInfo.set("Dataset updated")
        except:
            logging.exception('Error')
            self.statusInfo.set("Errors during processing, see the log")
            return

        # Refresh
        self.load_annotation(self.annoName, clear_log = False)
        self.statusInfo.set("Annotation updated: {}".format(self.annoName))

    def update_all_callback(self):
        self.update_img_size()

        GrLog.clear()
        try:
            generate_dataset(str(self.src_path), str(self.meta_path), str(self.img_path),
                str(self.sets_path), self.dsImgSize)
            if GrLog.numErrors() > 0:
                self.stoneInfo.set("Errors during processing, see the log")
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

    def load_annotation(self, file, clear_log = False):
        """Load annotation from file (XML/TXT)"""

        def get_tag(node, tag):
            d = node.getElementsByTagName(tag)
            if d is None: return None
            else:
                d = d[0].firstChild
                if d is None: return None
                else: return d.data

        def get_child_node(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        # Load annotation file
        if clear_log: GrLog.clear()
        try:
            with open(file) as f:
                data = minidom.parseString(f.read())

            # Find image file name
            fn = get_tag(data, 'path')
            if not Path(fn).is_file():
                fn = str(self.src_path.joinpath(Path(fn).name))
            logging.info('Image file {}'.format(fn))

            # Load image
            img = cv2.imread(fn)
            if img is None:
                raise Exception('File not found {}'.format(fn))

            # Resize the image
            logging.info('Image size {}'.format(img.shape[:2]))
            img2, self.zoom = resize2 (img, np.max(self.defBoardImg.shape[:2]), f_upsize = False)
            logging.info('Zooming to {}'.format(self.zoom))

            # Load objects list
            objs = data.getElementsByTagName('object')
            for ix, obj in enumerate(objs):
                x1 = int(get_child_node(obj, 'xmin'))
                y1 = int(get_child_node(obj, 'ymin'))
                x2 = int(get_child_node(obj, 'xmax'))
                y2 = int(get_child_node(obj, 'ymax'))
                cls = str(get_child_node(obj, "name")).lower().strip()
                logging.info('Class {} object ({},{}) - ({},{})'.format(cls, x1,y1, x2,y2))

                if x1 <= 0 or y1 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
                    logging.error("Point {} coordinates out of boundaries ({},{})-({},{}) <> ({},{})".format(ix, x1, \
                        y1,x2,y2,img.shape[1],img.shape[0]))
                if x1 >= x2 or y1 >= y2:
                    logging.error("Coordinates ({},{}) and ({},{}) overlap".format(x1,y1,x2,y2))

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
            self.imgFrame.pack_propagate(False)
            self.imgPanel.configure(image = self.boardImgTk)

            # Update status
            ds = self.find_dataset(self.boardImgName)
            if ds is None: ds = '<none>'
            img_info = "Size: ({}, {}), dataset: {}".format(img.shape[1], img.shape[0], ds)
            self.imgInfo.set(img_info)

            if GrLog.numErrors() > 0:
                self.stoneInfo.set("Errors during processing, see the log")
            else:
                self.statusInfo.set('File loaded: {}'.format(self.annoName))
        except:
            logging.exception('Error')
            self.statusInfo.set("Errors during processing, see the log")

    def update_img_size(self):
        try:
            n1 = int(self.trainSize.get())
            n2 = int(self.testSize.get())
            if n1 == 0 or n1 > 300:
                self.dsImgSize['train'] = n1
            else:
                raise Exception("Invalid value")
            if n2 == 0 or n2 > 300:
                self.dsImgSize['test'] = n2
            else:
                raise Exception("Invalid value")
            return True
        except:
            self.statusInfo.set("Image size should be integer equal to zero or greater than 300")
            return False

    def load_datasets(self):
        self.datasets = dict()
        for ds in ('test', 'train'):
            fn = self.sets_path.joinpath(ds + '.txt')
            try:
                with open(str(fn), 'r') as f:
                    self.datasets[ds] = f.read().splitlines()
                    f.close()
            except:
                self.datasets[ds] = []

    def save_datasets(self):
        for ds in ('test', 'train'):
            fn = self.sets_path.joinpath(ds + '.txt')
            try:
                with open(str(fn), 'w') as f:
                    f.writelines( "%s\n" % item for item in self.datasets[ds])
                    f.close()
            except:
                self.datasets[ds] = []


    def find_dataset(self, file):
        fn = Path(file).stem
        for ds in ('test', 'train'):
            try:
                if self.datasets[ds].index(fn) >= 0:
                    return ds
            except:
                pass
        return None

def main():
    window = tk.Tk()
    window.title("View annotaitons")

    log = GrLog.init()
    gui = ViewAnnoGui(window)

    window.mainloop()

if __name__ == '__main__':
    main()

