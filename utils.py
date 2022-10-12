import os
import shutil
import sys
import argparse
import json
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy import signal
import pickle


def mkdir(path,rm=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if rm:
            shutil.rmtree(path)
            os.mkdir(path)

class Logger(object) :
    def __init__ (self,filename='log.txt',stream=sys.stdout):
        self. terminal=stream
        self.log=open(filename,'a')

    def write(self,message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di