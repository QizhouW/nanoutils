import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy
import scipy.constants as sc
from scipy.signal import hilbert
import numpy.polynomial.polynomial as poly
import warnings
from prony_utils import prony, recons
import pickle
import pandas as pd
class Dataset():
    def __init__(self, grid_sizes, resdir, labels_dict_file, outputdir):
        self.grid_sizes = grid_sizes
        self.resdir = resdir
        self.outputdir = outputdir
        with open(labels_dict_file, 'rb') as f:
            self.label_dict = pickle.load(f)

    def load_data(self, data_csv):
        self.data = pd.read_csv(data_csv, index_col=0)

    def read_all(self):

    @staticmethod
    def merge_csv(list_of_data, outputfile):
        tmp = pd.concat(list_of_data).drop_duplicates(keep=False)
        tmp.to_csv(outputfile)
        print('Added data length: ', len(tmp))
        return True

    @staticmethod
    def clear_csv(all_data, parsed_data, outputfile):
        tmp = pd.concat([all_data, parsed_data, parsed_data]).drop_duplicates(keep=False)
        tmp.to_csv(outputfile)
        print('Clean data length: ', len(tmp))
        return True
