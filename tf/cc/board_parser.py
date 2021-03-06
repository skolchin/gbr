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
#from cc.board_areas import SampleBoardItemClassifier

class BoardParser:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dirs = { 'crossing': str(Path(root_dir).joinpath('cc', 'crossings')),
                          'stone': str(Path(root_dir).joinpath('cc', 'stones')),
                          'joined': str(Path(root_dir).joinpath('cc', 'joined'))
                        }

        self.model_dirs = { 'crossing': str(Path(root_dir).joinpath('tf', 'crossing_classifier')),
                            'stone': str(Path(root_dir).joinpath('tf', 'stone_classifier')),
                            'joined': str(Path(root_dir).joinpath('tf', 'joined_classifier'))}

        self.log_dir =  str(Path(root_dir).joinpath('tf', '_logs'))

        self.file_name = None
        self.n_train = None
        self.n_predict = None
        self.n_epochs = 20
        self.show_samples = False
        self.test_mode = False

        self.models = {}
        for k in self.model_dirs:
            self.models[k] = BoardItemClassifier(self.model_dirs[k], self.img_dirs[k], log_dir = self.log_dir)

    def arg_parse(self):
        model_names = [x for x in self.model_dirs]
        model_names.extend(['all'])

        parser = ArgumentParser()
        parser.add_argument("-n", "--name",
            help = 'Name of board image file')
        parser.add_argument('-i', "--show_samples",
            action = "store_true",
            help = 'Display some random images from all datasets')
        parser.add_argument('-t', "--train",
            choices = model_names,
            nargs = "*",
            help = 'Train one or several models')
        parser.add_argument('-p', "--predict",
            choices = model_names,
            nargs = "*",
            help = 'Does prediction on sample dataset images for one or several models')
        parser.add_argument('-e', "--epochs",
            type=int,
            default=20,
            help = 'Number of epochs in training')

##        parser.add_argument("--test",
##            action = "store_true",
##            help = 'Testing mode')

        args = parser.parse_args()
        self.file_name = args.name
        self.n_train = args.train
        self.n_predict = args.predict
        self.n_epochs = args.epochs
        self.show_samples = args.show_samples
        self.test_mode = False ##args.test

        if self.file_name is not None or self.n_train is not None or \
           self.show_samples or self.n_predict is not None:
            return True
        else:
            parser.print_help()
            return False

    def run(self):
        if not self.arg_parse():
            return

        if self.show_samples:
            # Show some random samples
            for k in self.models:
                self.models[k].init_datasets()
            for k in self.models:
                self.models[k].display_sample_images()

        if self.n_train is not None:
            # Retraing the models
            if len([k for k in self.n_train if k.lower() == 'all']) > 0:
                # Train all models
                for k in self.models:
                    self.train(k)
            else:
                # Train specifc models
                for k in self.n_train:
                    self.train(k)

        if self.n_predict is not None:
            # Predict the models
            if len([k for k in self.n_predict if k.lower() == 'all']) > 0:
                # Train all models
                for k in self.models:
                    self.predict(k)
            else:
                # Train specifc models
                for k in self.n_predict:
                    self.predict(k)

        if self.test_mode:
            self.test_edges()

    def train(self, name, display_history = False):
        self.models[name].init_datasets()
        self.models[name].train(epochs = self.n_epochs)
        self.models[name].save()
        if display_history:
            self.models[name].display_history()

    def predict(self, name):
        if not self.models[name].exists():
            print("Model '{}' does not exist".format(name))
        else:
            self.models[name].load()
            self.models[name].predict(display_predictions=True)

    def test_edges(self):
        pass
##        img_dir = str(Path(__file__).absolute().parent.parent)
##        self.edge_model = SampleBoardItemClassifier('crossing_classifier', img_dir)
##        self.edge_model.load()
##        self.edge_model.predict(num_samples = 1, display_predictions = True)

