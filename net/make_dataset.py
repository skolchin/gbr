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

def make_anno(meta_file, image_file, jgf):
    def stones_anno(f, jgf, cls):
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

    f = open(meta_file,'w')

    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t<filename>' + Path(image_file).name + '</filename>' + '\n'
    f.write(line)
    line = '\t<path>' + str(Path(image_file).absolute()) + '</path>' + '\n'
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

    stones_anno(f, jgf, 'black')
    stones_anno(f, jgf, 'white')

    line = "</annotation>" + '\n'
    f.write(line)
    f.close()

def main():

    root_path = "./gbr"
    src_path = "../img"
    meta_path = root_path + "/data/Annotations"
    img_path = root_path + "/data/Images"
    sets_path = root_path + "/data/ImageSets"

    # Create root and nested directories
    if not os.path.exists(meta_path):
       os.makedirs(meta_path)
    if not os.path.exists(img_path):
       os.makedirs(img_path)
    if not os.path.exists(sets_path):
       os.makedirs(sets_path)

    # Walk through the source board description files
    file_list = []
    src_p = Path(src_path)
    for file in os.listdir(src_path):
        if file.endswith('.jgf'):
           # Read into dictionary
           src_file = src_p.joinpath(file)
           file_list.append(str(src_file.with_suffix('').name))
           print ("Processing file {}".format(src_file))
           jgf = json.load(open(src_file))

           # Find image
           image_file = jgf.get("image_file")
           if image_file is None or image_file == "":
              image_file = str(src_file.with_suffix('.png'))

           # If the image file is PNG - copy it to Images dataset
           # Otherwise try to convert to PNG
           img_file = Path(image_file)
           png_file = Path(img_path).joinpath(img_file.name)
           if img_file.suffix == '.png':
              print("  {} -> {}".format(image_file, png_file))
              shutil.copy(image_file, png_file)
           else:
              png_file = str(Path(img_path).joinpath(img_file.with_suffix('.png').name))
              print("  {} -> {}".format(image_file, png_file))
              img = cv2.imread(image_file)
              cv2.imwrite(png_file, img)

           # Make annotation file
           meta_file = Path(meta_path).joinpath(img_file.with_suffix('.xml').name)
           make_anno(str(meta_file), str(png_file), jgf)

    # Save train.txt
    with open(sets_path + "/train.txt", "w+") as f:
         for file in file_list:
             f.write("{}\n".format(file))
         f.close()


if __name__ == '__main__':
    main()
