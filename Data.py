import os
import re

from torch import select

from Utils.CommonUtils import preprocess, expandDataset
from options_ import args
from Logger import Logger

class Data():
    def __init__(self, args, logger, rank):
        self.args = args
        self.logger = logger
        self.rank = rank

    def create_dataloader(self, base_path, test=False):
        loader_test = None
        sampler_test = None

        # get test_hdf5 path
        hdf5_path_test = os.path.join(base_path, 'benchmark', 'hdf5', self.args.data_test, 'X{}'.format(self.args.scale), 'test_database.hdf5')   

        if self.args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100', 'DIV2K']:
            if os.path.exists(hdf5_path_test):
                loader_test, sampler_test = preprocess(hdf5_path_test, 'test', self.args, test)
            else:
                if self.rank == 0:
                    self.logger.logger.info('does not exist test hdf5 file::{}'.format(hdf5_path_test))
                    self.logger.logger.info('make test hdf5 file::{}'.format(hdf5_path_test))
                self.make_hdf5_test(base_path)
                if self.rank == 0:
                    self.logger.logger.info('make test hdf5 file finished...')
                loader_test, sampler_test = preprocess(hdf5_path_test, 'test', self.args, test)
        return loader_test, sampler_test

    def make_hdf5_test(self, base_path):
        hr_path = os.path.join(base_path, 'benchmark', self.args.data_test, 'HR')
        lr_path = os.path.join(base_path, 'benchmark', self.args.data_test, 'LR_bicubic', 'X{}'.format(self.args.scale))
        expandDataset(hr_path,  lr_path, '', '', 64, 64, self.args.scale, 'test',  self.args.data_test, self.logger, self.rank, base_path)

if __name__ == "__main__":
    logger = Logger(args)
    data = Data(args, logger, 0)
    data.create_dataloader()
