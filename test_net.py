#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Deep-learning network testing module
#
# Author:      skolchin
#
# Created:     19.07.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import sys
import os
from pathlib import Path
import cv2
import caffe
import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

from gr.board import GrBoard
from gr.utils import img_to_imgtk, resize2
from gr.grdef import *
from gr.ui_extra import *
from gr.net_utils import cv2_show_detections

from PIL import Image, ImageTk

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog  as filedialog
    import ttk
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk

CLASSES = ["_back_", "white", "black"]

class GrTestNetGui(object):
    def __init__(self, root, max_size = 500, allow_open = True, num_iter = 10000):
        self.root = root
        self.allow_open = allow_open

        # Set paths
        self.root_path = Path(__file__).parent.resolve()
        self.model_file = self.root_path.joinpath("models", "test.prototxt")
        self.weigth_file = self.root_path.joinpath("out", "gbr_zf", "train", "gbr_zf_iter_" + str(num_iter) + ".caffemodel")

        # Top frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.buttonFrame = tk.Frame(self.root, width = max_size + 10, height = 70, bd = 1, relief = tk.RAISED)
        self.buttonFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.statusFrame = tk.Frame(self.root, width = max_size + 2*PADX, bd = 1, relief = tk.SUNKEN)
        self.statusFrame.pack(side = tk.BOTTOM, fill=tk.BOTH, padx = PADX, pady = PADY)

        # Image frame
        self.defBoardImg = GrBoard(board_shape = (max_size, max_size)).image
        self.boardImg = self.defBoardImg
        self.boardImgTk = img_to_imgtk(self.boardImg)
        self.boardImgName = None

        _, self.imgPanel, _ = addImagePanel(self.imgFrame,"DLN detection",
                [],
                self.boardImgTk, self.open_img_callback)

        # Buttons
        if self.allow_open:
            self.openBtn = tk.Button(self.buttonFrame, text = "Open",
                                                          command = self.open_btn_callback)
            self.openBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        self.updateBtn = tk.Button(self.buttonFrame, text = "Update",
                                                      command = self.update_callback)
        self.updateBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        # Params
        self.probFrame = tk.Frame(self.buttonFrame)
        self.probFrame.pack(side = tk.LEFT, padx = PADX, pady = PADY)
        self.netProb = 0.8

        tk.Label(self.probFrame, text = "Probability").grid(row = 0, column = 0)
        self.probVar = tk.StringVar()
        self.probVar.set(str(self.netProb))
        self.probEntry = tk.Entry(self.probFrame, textvariable = self.probVar)
        self.probEntry.grid(row = 0, column = 1)

        # Status frame
        self.statusInfo = tk.StringVar()
        self.statusInfo.set("")
        self.stoneInfoPanel = tk.Label(self.statusFrame, textvariable = self.statusInfo)
        self.stoneInfoPanel.grid(row = 0, column = 0, sticky = tk.W, padx = 5, pady = 2)


    def open_img_callback(self, event):
        self.open_btn_callback()

    def open_btn_callback(self):
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))
        if fn != "":
            self.load_image(fn)
            self.statusInfo.set("File loaded {}".format(fn))

    def update_callback(self):
        if not self.boardImgName is None:
            self.load_image(self.boardImgName)
            self.statusInfo.set("Detections updated")

    def load_image(self, file_name):
        # Load image
        img = cv2.imread(file_name)
        if img is None:
            raise Exception('File not found {}'.format(file_name))

        # Do detection and display results on the image
        self.update_prob()
        self.show_detection(img, self.netProb)

        # Resize the image
        img2, self.zoom = resize2 (img, np.max(self.defBoardImg.shape[:2]), f_upsize = False)

        # Display the image
        self.boardImg = img
        self.boardImgTk = img_to_imgtk(img)
        self.boardImgName = file_name
        self.imgFrame.pack_propagate(False)
        self.imgPanel.configure(image = self.boardImgTk)

    def show_detection(self, img, det_thresh):
        cfg.TEST.HAS_RPN = True
        cfg.TEST.BBOX_REG = False
        caffe.set_mode_gpu()

        net = caffe.Net(str(self.model_file), str(self.weigth_file), caffe.TEST)

##        print("== Network layers:")
##        for name, layer in zip(net._layer_names, net.layers):
##            print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
##
##        print("== Blobs:")
##        for name, blob in net.blobs.iteritems():
##            print("{:<5}:  {}".format(name, blob.data.shape))
##
##        img = cv2.imread(img_file)
##        scores, boxes = im_detect(net, img)
##
##        print("== Detections")
##        print("Scores: {}".format(scores))
##        print("Boxes: {}".format(boxes))

        # Detection
        scores, boxes = im_detect(net, img)

        # Draw results
        NMS_THRESH = 0.3
        colors = (0, (0,0,255), (255,0,0))
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            cv2_show_detections(img, cls, dets, thresh=det_thresh, f_label = False, color = colors[cls_ind])

    def update_prob(self):
        try:
            self.netProb = float(self.probVar.get())
        except:
            pass

def main():
    # Construct interface
    window = tk.Tk()
    window.title("View annotaitons")
    gui = GrTestNetGui(window)

    # Main loop
    window.mainloop()

if __name__ == '__main__':
    main()

