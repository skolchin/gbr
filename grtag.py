#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Main functions
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------
import numpy as np
import cv2
import sys
from PIL import Image, ImageTk
from pathlib import Path
import xml.dom.minidom as minidom
from gr.utils import img_to_imgtk, resize2
from gr.board import GrBoard

if sys.version_info[0] < 3:
    import Tkinter as tk
    import ttk
else:
    import tkinter as tk
    from tkinter import ttk


# GUI class
class GrTagGui:
      def __init__(self, root):
          self.root = root

          # Set paths
          self.root_path = Path(__file__).parent.resolve()
          self.src_path = self.root_path.joinpath("img")
          self.ds_path = self.root_path.joinpath("gbr_ds")
          if not self.ds_path.exists(): self.ds_path.mkdir(parents = True)
          self.meta_path = self.ds_path.joinpath("data","Annotations")
          if not self.meta_path.exists(): self.meta_path.mkdir(parents = True)
          self.img_path = self.ds_path.joinpath("data","Images")
          if not self.img_path.exists(): self.img_path.mkdir(parents = True)

          # File list panel
          self.filesFrame = tk.Frame(self.root)
          self.filesFrame.pack(side = tk.LEFT, fill=tk.BOTH, expand = True)

          # Files table
          self.fileListSb = tk.Scrollbar(self.filesFrame)
          self.fileListSb.pack(side=tk.RIGHT, fill=tk.BOTH)

          self.fileList = ttk.Treeview(self.filesFrame)
          self.fileList["columns"]=("json","jgf", "ds")
          self.fileList.column("#0", width=200, minwidth=100, stretch=tk.YES)
          self.fileList.column("json", width=50, minwidth=50, stretch=tk.YES)
          self.fileList.column("jgf", width=50, minwidth=50, stretch=tk.YES)
          self.fileList.column("ds", width=50, minwidth=50, stretch=tk.YES)
          self.fileList.heading("#0",text="Name",anchor=tk.W)
          self.fileList.heading("json", text="JSON",anchor=tk.W)
          self.fileList.heading("jgf", text="JGF",anchor=tk.W)
          self.fileList.heading("ds", text="DS",anchor=tk.W)
          self.load_files()
          self.fileList.pack(side = tk.TOP, fill=tk.BOTH, expand = True)

          self.fileListSb.config(command=self.fileList.yview)

      def load_files(self):
          def _add_file(name, file_list):
              file_list[x.name] = ( '1', '2', '3' )

          file_list = dict()
          for ext in ('*.png', '*.jpg'):
              g = self.src_path.glob(ext)
              for x in g:
                  if x.is_file(): _add_file(x, file_list)

          for f in sorted(file_list.keys()):
              self.fileList.insert("", "end", text = f, values=file_list[f])

def main():
    window = tk.Tk()
    window.title("Go tagging")

    gui = GrTagGui(window)

    #window.grid_columnconfigure(0, weight=1)
    #window.grid_rowconfigure(0, weight=1)
    #window.resizable(True, True)

    # Main loop
    window.mainloop()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
