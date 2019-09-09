#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Deep-learning network testing module
#
# Author:      skolchin
#
# Created:     19.07.2019
# Copyright:   (c) skolchin 2019
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
    def __init__(self, root, max_size = 500, allow_open = True):
        self.root = root
        self.allow_open = allow_open
        self.max_size = max_size

        # Set paths
        self.root_path = Path(__file__).parent.resolve()
        self.model_path = self.root_path.joinpath("models")
        self.model_file = 'zf_test.prototxt'
        self.weigth_path = self.root_path.joinpath("out", "gbr_zf", "train")
        self.weigth_file = None
        self.solver_file = 'zf_solver.prototxt'
        self.config_file = 'gbr_rcnn.yml'
        self.net_prob = 0.8
        self.net_iters = 10000

        # Top frames
        self.imgFrame = tk.Frame(self.root)
        self.imgFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.buttonFrame = tk.Frame(self.root, width = max_size + 10, height = 70, bd = 1, relief = tk.RAISED)
        self.buttonFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX, pady = PADY)
        self.configFrame = tk.Frame(self.root, bd = 1, relief = tk.RAISED)
        self.configFrame.pack(side = tk.TOP, fill=tk.BOTH, padx = PADX)
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

        self.trainBtn = tk.Button(self.buttonFrame, text = "Train",
                                                      command = self.train_callback)
        self.trainBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        self.updateBtn = tk.Button(self.buttonFrame, text = "Detect",
                                                      command = self.update_callback)
        self.updateBtn.pack(side = tk.LEFT, padx = PADX, pady = PADX)

        # Params
        self.probFrame = tk.Frame(self.configFrame)
        self.probFrame.pack(side = tk.LEFT, padx = PADX, pady = PADY)

        # Solver file
        self.solverVar, self.cbSolver = addField(self.probFrame, "cb", "Solver", 0, 0, self.solver_file)
        self.load_files(self.cbSolver, self.model_path, '*solver.prototxt')

        # Net train config
        self.configVar, self.cbConfig = addField(self.probFrame, "cb", "Config", 1, 0, self.config_file)
        self.load_files(self.cbConfig, self.model_path, '*.yml')

        # Number of iterations
        self.iterVar, self.iterEntry = addField(self.probFrame, "e", "Iterations", 2, 0, self.net_iters)

        # Model file
        self.modelVar, self.cbModel = addField(self.probFrame, "cb", "Model", 0, 2, self.model_file)
        self.load_files(self.cbModel, self.model_path, '*test.prototxt')

        # Weight file
        self.weigthVar, self.cbWeight = addField(self.probFrame, "cb", "Weights", 1, 2, self.weigth_file)
        self.load_files(self.cbWeight, self.weigth_path, '*.caffemodel')

        # Probability
        self.probVar, self.probEntry = addField(self.probFrame, "e", "Threshold", 2, 2, self.net_prob)

        # Status frame
        self.statusInfo = addStatusPanel(self.statusFrame, self.max_size + 10)
        self.statusInfo.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)


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

    def train_callback(self):
        # Set params
        if not self.update_train_params():
            return

        # Get args
        cmd = os.environ['FASTER_RCNN_HOME'] + '\\tools\\train_net.py'
        args = [sys.executable,
                cmd,
                '--solver',
                self.solver_file,
                '--imdb',
                'gbr_train',
                '--iters',
                str(self.net_iters),
                '--cfg',
                self.config_file]

        # Create a simple batch file
        with open("train.bat", "w") as f:
            f.write(':: This is an DLN training file generated by test_net.py module')
            f.write('@echo off\n')
            f.write('set PYTHONPATH=%PYTHONPATH%;.\n')
            f.writelines('%s ' % item for item in args)
            f.write('2>&1 | wtee out\\logs\\train.log\n')
        f.close()

        # Run the command
        os.system("start cmd.exe /k train.bat")


    def load_image(self, file_name):

        # Load image
        img = cv2.imread(file_name)
        if img is None:
            raise Exception('File not found {}'.format(file_name))

        # Update detections
        # Update params
        if self.update_net_params():
            self.show_detection(img, self.net_prob)

        # Resize the image
        img2, self.zoom = resize2 (img, np.max(self.defBoardImg.shape[:2]), f_upsize = False)

        # Display the image
        self.boardImg = img
        self.boardImgTk = img_to_imgtk(img2)
        self.boardImgName = file_name
        self.imgFrame.pack_propagate(False)
        self.imgPanel.configure(image = self.boardImgTk)

    def load_files(self, cb, path, mask):
        file_list = []
        g = path.glob(mask)
        for x in g:
          if x.is_file(): file_list.append(str(x.name))
        cb['values'] = sorted(file_list)

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

    def update_train_params(self):
        try:
            self.solver_file = self.get_file(self.model_path, self.solverVar)
            self.config_file = self.get_file(self.model_path, self.configVar)
            self.net_iters = int(self.get_entry(self.iterVar, 1000, 40000))
            return True
        except:
            self.statusInfo.set(str(sys.exc_info()[1]))
            return False

    def update_net_params(self):
        try:
            self.model_file = self.get_file(self.model_path, self.modelVar)
            self.weigth_file = self.get_file(self.weigth_path, self.weigthVar)
            self.net_prob = self.get_entry(self.probVar, 0.1, 100)
            return True
        except:
            self.statusInfo.set(str(sys.exc_info()[1]))
            return False

    def get_file(self, p, v):
        f = v.get()
        if f is None or f == '':
            raise ValueError('File not selected')
        return str(p.joinpath(f))

    def get_entry(self, v, min_, max_):
        f = float(v.get())
        if f < min_ or f > max_:
            raise ValueError('Value not in range {}, {}'.format(min_, max_))
        return f


def main():
    # Construct interface
    window = tk.Tk()
    window.title("View annotaitons")
    gui = GrTestNetGui(window)

    # Main loop
    window.mainloop()

if __name__ == '__main__':
    main()

