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
import gr
import grdef
import grutils

def main():
    img_path = Path("img").resolve()

    # Process all board parameters (JSON) files
    for file in os.listdir("img"):
        if file.endswith('.json'):
           # Load recognition parameters
           src_file = img_path.joinpath(file)
           print ("Processing file {}".format(src_file))

           grParams = grdef.DEF_GR_PARAMS.copy()
           p = json.load(open(str(src_file)))
           for key in grParams.keys():
               if p.get(key) is not None:
                  grParams[key] = p[key]
           print ("  Parameters loaded")

           # Load the file
           img_file = src_file.with_suffix('.png')
           if not img_file.exists():
              img_file = src_file.with_suffix('.jpg')
              if not img_file.exists():
                 print ("  Cannot find image file")
                 continue

           img = cv2.imread(str(img_file))
           grRes = gr.process_img(img, grParams)
           print ("  Image file processed: {}".format(img_file))

           # Update JGF
           jgf = grutils.gres_to_jgf(grRes)
           jgf['image_file'] = str(img_file.resolve())

           jgf_file = img_file.with_suffix('.jgf')
           with open(str(jgf_file), "w", encoding="utf-8", newline='\r\n') as f:
                json.dump(jgf, f, indent=4, sort_keys=True, ensure_ascii=False)

           print ("  JGF updated: {}".format(jgf_file))

if __name__ == '__main__':
    main()
