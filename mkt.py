#-------------------------------------------------------------------------------
# Name:        Transparent image
# Purpose:     Small script to make transparent images
#              Origin https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent
#
# Author:      kol
#
# Created:     18.11.2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from PIL import Image

def transparent(src_file, dest_file, transp_color = (255, 255, 255)):
    img = Image.open(src_file)
    img = img.convert("RGBA")

    check_color = transp_color + (255, )
    replace_color = transp_color + (0, )

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == check_color:
                pixdata[x, y] = replace_color

    img.save(dest_file, "PNG")

def main():
    fs = "C:/Users/kol/Documents/kol/gbr/ui/save_flat.png"
    fd = "C:/Users/kol/Documents/kol/gbr/ui/save_flat2.png"
    transparent(fs, fd, (192, 192, 192))

if __name__ == '__main__':
    main()
