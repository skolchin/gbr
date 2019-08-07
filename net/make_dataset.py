#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Make a dataset to train the net
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
import shutil

def annotate_stones(f, jgf, cls):
    stones = jgf[cls]
    for i in stones:
        line = '\n\t<object>'
        line += '\n\t\t<name>' + cls + '</name>\n\t\t<pose>Unspecified</pose>'
        line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>Unspecified</difficult>'

        x = stones[i]['X']
        y = stones[i]['Y']
        r = stones[i]['R']

        xmin = x - int(r/2)
        ymin = y - int(r/2)
        xmax = x + int(r/2)
        ymax = y + int(r/2)

        line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
        line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
        line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
        line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
        line += '\n\t\t</bndbox>'

        line += '\n\t</object>'

        f.write(line)

def make_anno(meta_file, image_file, jgf = None):
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

    if not jgf is None:
        annotate_stones(f, jgf, 'black')
        annotate_stones(f, jgf, 'white')

    line = "\n</annotation>" + '\n'
    f.write(line)
    f.close()

def resize(img, max_size):
    h, w = img.shape[:2]
    z = [1.0, 1.0]
    if (w > max_size):
        z[0] = float(max_size) / float(w)
        z[1] = z[0]
    elif (h > max_size):
        z[1] = float(max_size) / float(h)
        z[0] = z[1]
    else:
        if w >= h:
            z[0] = float(max_size) / float(w)
            z[1] = z[0]
        else:
            z[1] = float(max_size) / float(h)
            z[0] = z[1]
    img2 = cv2.resize(img, dsize = None, fx = z[0], fy = z[1])
    return img2

def main():

    root_path = Path(__file__).with_name('').joinpath('..').resolve()
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
            make_anno(str(meta_file), str(png_file), jgf)

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
            img2 = resize(img, 300)
            cv2.imwrite(str(png_file), img2)
            print("  {} -> {}".format(src_file, png_file))

            # Make annotation file
            meta_file = meta_path.joinpath(src_file.with_suffix('.xml').name)
            make_anno(str(meta_file), str(png_file))

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

if __name__ == '__main__':
    main()

