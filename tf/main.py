#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     17.01.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from pathlib import Path
from cc.board_parser import BoardParser

parser = None

def main():
    global parser

    root_dir = str(Path(__file__).absolute().parent.parent)
    print('==> Root directory is', root_dir)

    parser = BoardParser(root_dir)
    parser.run()

if __name__ == '__main__':
    main()
