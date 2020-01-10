#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     22.12.2019
# Copyright:   (c) kol 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

#import sys
#sys.path.append('../')
#from gr.cv2_watershed import apply_watershed

class App(tk.Tk):
    def get_cascade_fn(self):
        return filedialog.askopenfilename(
            title = "Select cascade file",
            initialdir = "m/",
            initialfile = "cascade.xml",
            filetypes = (("XML","*.xml"), ("All files","*.*")))

    def get_image_fn(self):
        return filedialog.askopenfilename(
            title = "Select image file",
            initialdir = "../img",
            initialfile = "go_board_1.png",
            filetypes = (("PNG files","*.png"),("JPEG files","*.jpg"),("All files","*.*")))

    def main(self):
        cascade_fn = self.get_cascade_fn()
        if cascade_fn == '': return
        cascade = cv2.CascadeClassifier(cascade_fn)

        while True:
            img_fn = self.get_image_fn()
            if img_fn == '': return

            img = cv2.imread(img_fn)
            if img is None:
                print("Not found")
            else:
                max_x = int((img.shape[1] - 20) / 19) * 2
                max_y = int((img.shape[0] - 20) / 19) * 2
                print("Maximum rect size {}".format((max_x, max_y)))

                results = cascade.detectMultiScale(img, scaleFactor = 1.1, maxSize = (max_x, max_y))
                if len(results) == 0:
                    print('No stones found')
                else:
                    results[:,2:] += results[:,:2]
                    print(results)

                    stones = []
                    for r in results:
                        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), color = (0, 0, 255))
                        mx = int((r[0] + r[2])/2)
                        my = int((r[1] + r[3])/2)
                        stones.extend([[mx, my]])


                    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #stones_ws, debug = apply_watershed(gray, np.array(stones), 150, 'W', f_debug = True)

                cv2.imshow('Results', img)
                cv2.waitKey()

def main():
    app = App()
    app.main()
    app.destroy()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
