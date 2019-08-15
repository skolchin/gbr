#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Update JGF files for all images with board parameters saved
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------
import os
from pathlib import Path
import cv2
import json
import numpy as np
from gr.board import GrBoard

def main():
    img_path = Path("img").resolve()

    # Process all board parameters (JSON) files
    cnt = 0
    for file in os.listdir("img"):
        if file.endswith('.json'):
           # Load recognition parameters
           src_file = img_path.joinpath(file)
           print ("Processing file {}".format(src_file))

           img_file = src_file.with_suffix('.png')
           if not img_file.exists():
              img_file = src_file.with_suffix('.jpg')
              if not img_file.exists():
                 print ("  Cannot find image file")
                 continue

           # Load the file
           board = GrBoard(str(img_file))

           # Update JGF
           jgf_file = board.save_params()
           print ("  JGF updated: {}".format(jgf_file))
           cnt += 1

    print("File(s) processed: {}".format(cnt))

if __name__ == '__main__':
    main()
