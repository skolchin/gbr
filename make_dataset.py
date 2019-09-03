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
import logging
from gr.board import GrBoard

DEF_DS_IMG_SIZE =  { 'test': 0, 'train': 2048 }
DS_FORMAT_PASCAL = "pascal"

def default_image_size():
    return DEF_DS_IMG_SIZE

def generate_dataset(src_path, meta_path, img_path, sets_path, img_size = DEF_DS_IMG_SIZE, format = DS_FORMAT_PASCAL):
    # Prepare file list
    file_list = dict()
    file_list['test'] = []
    file_list['train'] = []

    # Process all JGF (board descr) files in source path
    for file in os.listdir(src_path):
        src_file = os.path.join(src_path, file)
        board = GrBoard()
        stage = ""
        max_size = None

        if file.endswith('.jgf'):
            # Process JGF file. Images with JGF would go training dataset
            board.load_board_info(src_file, f_use_gen_img = False, path_override = src_path)
            stage = "train"
        elif file.endswith('.jpg') or file.endswith('.png'):
            # Find JGF. Images without board info will be used in testing dataset
            jgf_file = os.path.splitext(src_file)[0] + '.jgf'
            if os.path.exists(jgf_file): continue

            # Process image file
            logging.info ("Loading file {} to testing dataset".format(src_file))
            board.load_image(src_file, f_process = False)
            stage = "test"
        else:
            continue

        logging.info ("Appending to dataset {}".format(stage))

        # Convert to PNG
        image_file = os.path.basename(board.image_file)
        png_file = os.path.splitext(os.path.join(img_path,image_file))[0] + '.png'
        if not img_size[stage] is None and img_size[stage] > 0:
            board.resize_board(img_size[stage])
        board.save_image(str(png_file))

        # Save annotation
        meta_file = os.path.splitext(os.path.join(meta_path,image_file))[0] + '.xml'
        board.save_annotation(meta_file)

        # Add to file list
        file_list[stage].append(os.path.splitext(os.path.basename(png_file))[0])

    # Save datasets
    for mode in file_list:
        ds_name = os.path.join(sets_path, mode+'.txt')
        logging.info("Creating dataset {}".format(ds_name))
        count = 0
        with open(str(ds_name), "w+") as f:
            for file in file_list[mode]:
                f.write("{}\n".format(file))
                count += 1
            f.close()
        logging.info("{} entries written".format(count))

def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level = logging.INFO)

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

    generate_dataset(str(src_path), str(meta_path), str(img_path), str(sets_path))

if __name__ == '__main__':
    main()

