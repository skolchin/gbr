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
import os
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from cc.item_classifier import BoardItemClassifier

class BoardParser:
    def __init__(self):
        self.img_dirs = { 'crossing': str(Path(__file__).absolute().parent.parent.joinpath('cc', 'crossings')),
                          'stone': str(Path(__file__).absolute().parent.parent.joinpath('cc', 'stones'))}

        self.model_dirs = { 'crossing': str(Path(__file__).absolute().parent.joinpath('crossing_classifier')),
                            'stone': str(Path(__file__).absolute().parent.joinpath('stone_classifier'))}
        self.log_dir =  str(Path(__file__).absolute().parent.joinpath('_logs'))

        self.file_name = None
        self.n_train = None
        self.n_epochs = 20

        self.models = {}
        for k in self.model_dirs:
            self.models[k] = BoardItemClassifier(self.model_dirs[k], self.img_dirs[k], log_dir = self.log_dir)

    def arg_parse(self):
        parser = ArgumentParser()
        parser.add_argument("file_name",
            help = 'Name of board image file')
        parser.add_argument('-t', "--train",
            choices = ["stone", "crossing", "all"],
            nargs = "*",
            help = 'Train one or several models (valid names: stone, crossing or all)')
        parser.add_argument('-n', "--epochs",
            type=int,
            default=20,
            help = 'Number of epochs in training')

        args = parser.parse_args()
        self.file_name = args.file_name
        self.n_train = args.train
        self.n_epochs = args.epochs

    def run(self):
        self.arg_parse()

        if self.n_train is not None:
            if len([k for k in self.n_train if k.lower() == 'all']) > 0:
                # Train all models
                for k in self.models:
                    self.train(k)
            else:
                # Train specifc models
                for k in self.n_train:
                    self.train(k)

    def train(self, name, display_history = False):
        self.models[name].init_datasets()
        self.models[name].train(epochs = self.n_epochs)
        self.models[name].save()
        if display_history:
            self.models[name].display_history()

    def process(self):
        #classifier.load()
        #classifier.predict(64)
        pass

parser = None

def main():
    global parser
    parser = BoardParser()
    parser.run()

if __name__ == '__main__':
    main()
