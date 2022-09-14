import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy
import scipy.constants as sc
from scipy.signal import hilbert
import numpy.polynomial.polynomial as poly
import warnings


def prony(x,m,dt,f_ls,downsample_factor,dt_normalize,amp_normalize,sort=False,iscos=True):
    #iscos=False   # see if the passed function is cosine based, if so, double the number of modes
    # Polynomial Fitting Method
    #x=x[::10]
    #x=x[200:]
    x=x/amp_normalize
    if iscos:
        p=2*m
    else:
        p=m

    system_order=np.min([3*p,len(x)])   # betwwen p (exact) and N (overdetermined), but can not be too big
    
    f_ls=f_ls*downsample_factor/dt_normalize
    dt=dt*dt_normalize
    f_ls=np.concatenate([-1*f_ls,f_ls])
    j=0+1j
    r=np.exp(j*f_ls*np.pi*2*dt)
    # S3
    Z=np.empty([system_order,p],dtype=complex)
    for i in range(system_order):
        Z[i,:]=np.power(r,i+1,dtype=complex)    # The terms matching has a little error, need to use Z_k^i+1 
    x_hat=x[0:system_order+0]
    H=np.dot(np.linalg.pinv(Z),x_hat)

    # S4
    alpha=np.log(np.abs(r))/dt 
    freq=np.angle(r)/dt/2 /np.pi/downsample_factor*dt_normalize
    amp=np.abs(H)*amp_normalize
    phi=np.angle(H)
    
    if sort:  # remove negative freq, note that this result cannot be use for reconstruction
        positive_index=np.where(freq>0)
        freq=freq[positive_index]
        alpha=alpha[positive_index]
        amp=amp[positive_index]
        phi=phi[positive_index]

        sort_index=np.argsort(freq)
        freq=freq[sort_index]
        alpha=alpha[sort_index]
        amp=amp[sort_index]
        phi=phi[sort_index]
    
    return np.stack([alpha,amp,freq,phi],axis=0)




def recons(params,x,dt,vis,plotlim=None):
    if plotlim is None:
        plotlim=len(x)-1
    t=np.arange(1,len(x)+1)*dt 
    res=0
    p=params.shape[1]
    for i in range(p):
        w=params[2,i]*2*np.pi
        phase=params[3,i]
        alp=params[0,i]
        amp=params[1,i]
        res=res+amp*np.exp((0+1j)*(w*t+phase))*np.exp(-alp*t)
    err=np.mean(np.abs(x-res)/np.mean(np.abs(x)))**2
    if vis:
        
        print('Extracted freq: ',params[2])
        #print('Real freq: +-', np.linspace(f1,f2,m))
        print('Extracted amplitude: ',params[1])
        #print('Real amplitude: 1')
        print('Extracted phase: ',params[3])
        #print('Real phase:',1)
        print('Extracted dumping coefficient: ',params[0])
        #print('Real dumping coefficient: 0')
        plt.plot(t,res)
        plt.plot(t,x)
        plt.xlim(t[0],t[plotlim])
        plt.legend(['reconstructed','origin'])
        #plt.show()
    return err
