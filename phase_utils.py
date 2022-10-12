import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy
import scipy.constants as sc
from scipy.signal import hilbert
import numpy.polynomial.polynomial as poly
import warnings
from prony_utils import prony,recons


def parse_phase(data,m,wl1,wl2,D,dt,skip=10000):
    # Return the recons error and phase, amplitude for each CW componet
    f1=sc.c/wl1
    f2=sc.c/wl2
    f_ls=np.linspace(f1,f2,m)
    data=data[skip:]
    x_down=data[D-1::D]
    x_train=x_down[:300]
    params=prony(x_train,m,dt,f_ls,iscos=True,downsample_factor=D,dt_normalize=1e9,amp_normalize=1)
    mse=recons(params,data,dt,vis=False,showparam=False,plotlim=None)
    t=np.arange(1,len(data)+1)*dt 
    td=t[D-1::D][:300]
    p=prony(x_train,m,dt,f_ls,iscos=True,downsample_factor=D,dt_normalize=1e9,amp_normalize=1,sort=True)
    return mse, p[3], p[1]

def diff_phase(psrc,pful,unwrap=False):
    dp=psrc-pful
    dp=dp%(np.pi*2)
    if unwrap:
        dp=np.unwrap(dp)
    return dp
    
def extract_wl(ex,wl1,wl2,D,m,dt=4.81458e-12,skip=10000):
    f1=sc.c/wl1
    f2=sc.c/wl2
    f_ls=np.linspace(f1,f2,m)
    ex=ex[skip:]
    x_down=ex[D-1::D]
    x_train=x_down[:300]
    p=prony(x_train,m,dt,f_ls,iscos=True,downsample_factor=D,dt_normalize=1e9,amp_normalize=1,sort=True)
    wl=sc.c/p[2]
    return np.array(wl)

