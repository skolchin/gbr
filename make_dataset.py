#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network dataset generation
#
# Author:      skolchin
#
# Created:     03.08.2019.
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from pathlib import Path
import cv2
import json
import numpy as np
import math
import os
from gr.board import GrBoard

MAX_SIZE = 300

def main():
    root_path = Path(__file__).parent.resolve()
    ds_path = root_path.joinpath("gbr_ds")
    if not ds_path.exists(): ds_path.mkdir(parents = True)

    src_path = root_path.joinpath("img")
    meta_path = ds_path.joinpath("data\\Annotations")
    if not meta_path.exists(): meta_path.mkdir(parents = True)
    img_path = ds_path.joinpath("data\\Images")
    if not img_path.exists(): img_path.mkdir(parents = True)
    sets_path = ds_path.joinpath("data\\ImageSets")
    if not sets_path.exists(): sets_path.mkdir(parents = True)

    # Prepare file list
    file_list = dict()
    file_list['test'] = []
    file_list['train'] = []

    # Process all JGF (board descr) files
    for file in os.listdir(str(src_path)):
        src_file = src_path.joinpath(file)
        board = GrBoard()
        stage = ""
        max_size = None

        if file.endswith('.jgf'):
            # Process JGF file
            print ("Processing file {}".format(src_file))
            board.load_board_info(str(src_file), f_use_gen_img = False, path_override = str(src_path))
            stage = "train"
        elif file.endswith('.jpg') or file.endswith('.png'):
            # Find JGF
            jgf_file = src_path.joinpath(file).with_suffix('.jgf')
            if jgf_file.exists(): continue

            # Process image file
            print ("Processing file {}".format(src_file))
            if not src_file.is_file():
               src_file = src_path.joinpath(src_file.name)
            board.load_image(str(src_file), f_process = False)
            stage = "test"
            max_size = MAX_SIZE
        else:
            continue

        # Convert to PNG
        image_file = Path(board.image_file).name
        png_file = img_path.joinpath(image_file).with_suffix('.png')
        print("  {} -> {}".format(image_file, png_file))
        board.save_image(str(png_file), max_size)

        # Save annotation
        meta_file = meta_path.joinpath(image_file).with_suffix('.xml')
        board.save_annotation(meta_file)

        # Add to file list
        file_list[stage].append(str(png_file.stem))

    # Save datasets
    for mode in file_list:
        ds_name = sets_path.joinpath(mode+'.txt')
        print("Creating dataset {}".format(ds_name))
        count = 0
        with open(str(ds_name), "w+") as f:
            for file in file_list[mode]:
                f.write("{}\n".format(file))
                count += 1
            f.close()
        print("{} entries written".format(count))


if __name__ == '__main__':
    main()

