#-------------------------------------------------------------------------------
# Name:        Go board recognition
# Purpose:     Script to create a dataset
#
# Author:      kol
#
# Created:     06-09-2019
# Copyright:   (c) kol 2019
# Licence:     MIT
#-------------------------------------------------------------------------------
from pathlib import Path
import logging
from gr.dataset import GrDataset


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level = logging.INFO)

    ds = GrDataset.getDataset()
    ds.use_image_ids = True
    ds.separate_stages = True
    ds.generate_dataset()

if __name__ == '__main__':
    main()


