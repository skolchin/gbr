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
from gr.utils import img_to_imgtk, resize

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    from tkinter import ttk

sys.path.append('..')
from gr.utils import resize, img_to_imgtk

class ViewAnnoGui:
      def __init__(self, root):
          self.root = root

          # Set paths
          self.root_path = Path(__file__).parent.resolve()
          self.ds_path = self.root_path.joinpath("gbr_ds")
          if not self.ds_path.exists(): self.ds_path.mkdir(parents = True)
          self.meta_path = self.ds_path.joinpath("data","Annotations")
          self.img_path = self.ds_path.joinpath("data","Images")

          # File list panel
          self.filesFrame = tk.Frame(self.root)
          self.filesFrame.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = "nswe")

          # File list listbox
          self.annoFileName = ''
          self.fileListSb = tk.Scrollbar(self.filesFrame)
          self.fileListSb.pack(side=tk.RIGHT, fill=tk.BOTH)
          self.fileList = tk.Listbox(self.filesFrame, yscrollcommand=self.fileListSb.set)
          self.fileList.pack(side = tk.TOP, fill=tk.BOTH, expand = True)
          self.fileListSb.config(command=self.fileList.yview)
          self.load_files()
          self.fileList.bind("<<ListboxSelect>>", self.lb_changed_callback)

          # Image frame
          self.imgFrame = tk.Frame(self.root)
          self.imgFrame.grid(row = 0, column = 1, padx = 5, pady = 5, sticky = "nswe")
          self.defBoardImg = cv2.imread('def_board.png')
          self.boardImg = self.defBoardImg
          self.boardImgTk = img_to_imgtk(self.boardImg)
          self.boardImgName = ''
          self.imgPanel = tk.Label(self.imgFrame, image = self.boardImgTk)
          self.imgPanel.pack(fill=tk.BOTH, anchor='center', expand=True, padx = 3, pady = 3)

          # Buttons on image frame
          self.buttonFrame = tk.Frame(self.imgFrame, bd = 1)
          self.buttonFrame.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True, padx = 3, pady = 3)

          self.updateBtn = tk.Button(self.buttonFrame, text = "Update",
                                                        command = self.update_callback)
          self.updateBtn.pack(side = tk.LEFT, padx = 5, pady = 5)

          self.openBtn = tk.Button(self.buttonFrame, text = "Open in GBR",
                                                        command = self.open_callback)
          self.openBtn.pack(side = tk.LEFT, padx = 5, pady = 5)

      def load_files(self):
          g = self.meta_path.glob('*.xml')
          for x in g:
              if x.is_file(): self.fileList.insert(tk.END, x.name)

      def lb_changed_callback(self, event):
          w = event.widget
          index = int(w.curselection()[0])
          file = w.get(index)
          self.load_anno(file)

      def load_anno(self, file):

          def get_data_from_tag(node, tag):
                return node.getElementsByTagName(tag)[0].childNodes[0].data

          # Load annotation file
          print("Loading annotation {}".format(file))
          fn = str(self.meta_path.joinpath(file))
          with open(fn) as f:
            data = minidom.parseString(f.read())
          self.annoFileName = fn

          # Find image file name
          fn_list = data.getElementsByTagName('path')
          if fn_list is None or len(fn_list) == 0:
             raise Exception('Filename not specified')

          fn = fn_list[0].firstChild.data
          print("Loading image {}".format(fn))

          # Load and resize the image
          img = cv2.imread(fn)
          if img is None:
             raise Exception('File not found')
          img2 = resize(img, np.max(self.defBoardImg.shape[:2]), f_upsize = False)

          # Load objects list
          objs = data.getElementsByTagName('object')
          for ix, obj in enumerate(objs):
              x1 = int(get_data_from_tag(obj, 'xmin'))
              y1 = int(get_data_from_tag(obj, 'ymin'))
              x2 = int(get_data_from_tag(obj, 'xmax'))
              y2 = int(get_data_from_tag(obj, 'ymax'))
              cls = str(get_data_from_tag(obj, "name")).lower().strip()
              if x1 <= 0 or y1 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
                print("ERROR: coordinate out of boundary")
              if x1 >= x2 or y1 >= y2:
                print("ERROR: coordinates overlap")

              # Draw a bounding box
              cv2.rectangle(img2,(x1,y1),(x2,y2),(0,255,0),2)
              d = max(x2-x1, y2-y1)
              x = int(x1 + d/2)
              y = int(y1 + d/2)
              #cv2.circle(img2, (x,y), int(d/2), (0,0,255), 1)

          # Display the image
          self.boardImg = img2
          self.boardImgTk = img_to_imgtk(img2)
          self.boardImgName = fn
          self.imgFrame.pack_propagate(False)
          self.imgPanel.configure(image = self.boardImgTk)

      def update_callback(self):
          #make_anno(self.annoFileName, self.boardImgName)
          pass

      def open_callback(self):
          pass


def main():
    # Construct interface
    window = tk.Tk()
    window.title("View annotaitons")
    gui = ViewAnnoGui(window)

    # Main loop
    window.mainloop()

if __name__ == '__main__':

    main()
