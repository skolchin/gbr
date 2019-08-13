#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Deep learning network dataset generation
#
# Author:      skolchin
#
# Created:     03.08.2019
# Copyright:   (c) skolchin 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
from pathlib import Path
import cv2
import json
import numpy as np
from net_utils import show_detections
from matplotlib import pyplot as plt
import math

MAX_SIZE = 300
VISUALIZE = True

def annotate_stones(f, jgf, cls):
    stones = jgf[cls]
    bbox = np.empty((len(stones),5), dtype = np.float)

    n = 0
    for i in stones:
        x = stones[i]['X']
        y = stones[i]['Y']
        r = stones[i]['R']
        if r < 4: continue   # Skip objects too small

        a = 2 * r * math.sqrt(2)
        xmin = x - int(a/2)
        ymin = y - int(a/2)
        xmax = x + int(a/2)
        ymax = y + int(a/2)

        bbox[n,0] = xmin
        bbox[n,1] = ymin
        bbox[n,2] = xmax
        bbox[n,3] = ymax
        bbox[n,4] = 100.0
        n += 1

        line = '\n\t<object>'
        line += '\n\t\t<name>' + cls + '</name>\n\t\t<pose>Unspecified</pose>'
        line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>Unspecified</difficult>'

        line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
        line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
        line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
        line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
        line += '\n\t\t</bndbox>'

        line += '\n\t</object>'

        f.write(line)

    return bbox

def make_anno(meta_file, image_file, jgf = None, f_vis = False):
    f = open(meta_file,'w')

    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t<filename>' + Path(image_file).name + '</filename>' + '\n'
    f.write(line)
    line = '\t<path>' + str(Path(image_file).resolve()) + '</path>' + '\n'
    f.write(line)
    line = '\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
    f.write(line)

    im = cv2.imread(image_file)
    (height, width, depth) = im.shape
    line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
    line += '\t<depth>' + str(depth) + '</depth>\n\t</size>'
    f.write(line)
    line = '\n\t<segmented>Unspecified</segmented>'
    f.write(line)

    bb_b = None
    bb_w = None
    if not jgf is None:
        bb_b = annotate_stones(f, jgf, 'black')
        bb_w = annotate_stones(f, jgf, 'white')

    line = "\n</annotation>" + '\n'
    f.write(line)
    f.close()

    if f_vis and not bb_b is None:
       title = "Showing {} from {}".format("black", image_file)
       show_detections(im, "black", bb_b, 0.0,
                           f_label = False, f_title = True, title = title, ptype = "r")
    if f_vis and not bb_w is None:
       title = "Showing {} from {}".format("white", image_file)
       show_detections(im, "white", bb_w, 0.0,
                           f_label = False, f_title = True, title = title, ptype = "r")

def resize(img, max_size):
    im_size_max = np.max(img.shape[0:2])
    im_size_min = np.min(img.shape[0:2])
    im_scale = float(max_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    img2 = cv2.resize(img, dsize = None, fx = im_scale, fy = im_scale)
    return img2

def main():
    root_path = Path(__file__).parent.joinpath('..').resolve()
    ds_path = root_path.joinpath("net", "gbr_ds")
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
    n_jgf = 0
    n_img = 0
    n_vis = 0
    for file in os.listdir(str(src_path)):
        src_file = src_path.joinpath(file)
        print ("Processing file {}".format(src_file))
        if file.endswith('.jgf'):
            # Process JGF file
            jgf = json.load(open(str(src_file)))

            # Find image tag
            image_file = jgf.get("image_file")
            if image_file is None or image_file == "":
                print("   ERROR: cannot find file {}".format(img_file))
                continue

            # Check image file
            img_file = Path(image_file)
            if not img_file.exists():
                # W/A to consider root directory changes among diff installations
                img_file = Path(src_path).joinpath(img_file.name)
            if not img_file.exists():
                print("   ERROR: cannot find file {}".format(img_file))
                continue

            # Convert to PNG
            png_file = img_path.joinpath(img_file.name)
            if png_file.suffix != '.png':
                png_file = png_file.with_suffix('.png')

            img = cv2.imread(str(img_file))
            cv2.imwrite(str(png_file), img)
            print("  {} -> {}".format(image_file, png_file))

            # Make annotation file
            meta_file = meta_path.joinpath(img_file.with_suffix('.xml').name)
            f_vis = VISUALIZE and (n_vis <= 10)
            make_anno(str(meta_file), str(png_file), jgf = jgf, f_vis = f_vis)
            if f_vis: n_vis += 1
            n_jgf += 1

            # Add to file list
            file_list["train"].append(str(png_file.stem))

        elif file.endswith('.jpg') or file.endswith('.png'):
            # Find JGF
            jgf_file = src_path.joinpath(file).with_suffix('.jgf')
            if jgf_file.exists():
                print ("  JGF file found, skipping image")
                continue

            # Convert to PNG
            png_file = img_path.joinpath(src_file.name)
            if png_file.suffix != '.png':
                png_file = png_file.with_suffix('.png')

            img = cv2.imread(str(src_file))
            img2 = resize(img, MAX_SIZE)
            cv2.imwrite(str(png_file), img2)
            print("  {} -> {}".format(src_file, png_file))

            # Make annotation file
            meta_file = meta_path.joinpath(src_file.with_suffix('.xml').name)
            make_anno(str(meta_file), str(png_file))
            n_img += 1

            # Add to file list
            file_list["test"].append(str(png_file.stem))
        else:
            print ("  Unknown extension, skipping")

    # Save datasets
    for mode in file_list:
        ds_name = sets_path.joinpath(mode+'.txt')
        print("Creating dataset {}".format(ds_name))
        with open(str(ds_name), "w+") as f:
            for file in file_list[mode]:
                f.write("{}\n".format(file))
            f.close()

    print("File(s) processed: {} boards, {} images".format(n_jgf, n_img))
    if n_vis > 0: plt.show()

if __name__ == '__main__':
    main()

