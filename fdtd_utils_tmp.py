import numpy as np
from scipy import linalg as la
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import scipy.io as sio
import math
from scipy.linalg import hankel
from scipy.linalg import pinv
from scipy.linalg import eig
from scipy.optimize import minimize
#%matplotlib                                      
from pyevtk.hl import imageToVTK          
import scipy.constants as sc
import time
import os
import sys
from scipy.signal import residue

########################### MISC FUNCTIONS                                                                                                                
##### get power                                                                                                       
def get_p(dat,ts_p,nrm):
    hx=dat[::4]
    ex=dat[1::4]
    ey=dat[2::4]
    hy=dat[3::4]
    pw=-ey*hx+hy*ex
    pwt=pw[-2*ts_p:-ts_p]
    return np.abs(np.mean(pwt))*nrm,pw*nrm

####unpack TDCMT/HARM data
####modes are returned in list of xz (00,01,02,...,0N,10,11,...,MN)            
def unpack(file,tsteps):
    data=np.fromfile(file,dtype=np.float64)
    nmodes=int(len(data)/(tsteps+1))
    data=data.reshape(tsteps+1,nmodes)
    return data


####### TDCMT CLASS
class tdcmt(object):

    #private
    def fit_best(self, mode, err_th=1, fmax=10, max_N=3):
        thresholds = np.arange(0,0.5,0.1)
        lstsq_err = 1e10
        for N in range(1, max_N+1):
            for th in thresholds:
                fit_err, lstsq_err = self.fitone(mode, th, fmax, N, plot=False)
                if lstsq_err <= err_th:
                    return self.a, lstsq_err

        return None, lstsq_err

    def fit_given_modes(self, modes_idxs, rewrite = True, *pargs, **kargs):
        if not rewrite:
            self.models = np.zeros(self.nmodes, dtype=object)
        errs = np.zeros(self.nmodes, dtype=float)
        for i in modes_idxs:
            self.models[i], errs[i] = self.fit_best(i, *pargs, **kargs)
        fails = np.where(self.models == None)
        return errs[:], fails


    def fit_all(self, *pargs, **kargs):
        self.models = np.zeros(self.nmodes, dtype=object)
        errs = np.zeros(self.nmodes, dtype=float)
        for i in range(self.nmodes):
            self.models[i], errs[i] = self.fit_best(i,*pargs,**kargs)

        fails = np.where(self.models == None)
        return errs, fails[0]

    def get_lstsq_error(self, w, fun, fit):
        return np.sum(np.abs(fit-fun)**2)
    # fit the spectrum of mode id from th*max to max below fmax with minimum number of poles N
    def fitone(self,id,th,fmax, N = 3, plot=True):
        ff=self.f<=fmax
        max=np.max(np.abs(self.smode[ff,id]))
        msk=(np.abs(self.smode[:,id])>=th*max)*(self.f<=fmax)#(self.f>=f0)*(self.f<=f1)
        self.msk = msk
        w=self.f[msk]
        fun=self.smode[msk,id]
        a=frac_fit()
        a.fit0(w,fun,np.real(fun[0]),N)
        fted=a.out(w)
        self.err = a.sol[1]
        self.a = a
        lstsq_err= self.get_lstsq_error(ff, self.smode[ff,id], a.out(self.f[ff]))
        if plot and lstsq_err < 1e2:
            print(self.err, lstsq_err, id)
            plt.figure(1)
            #plt.clf()
            plt.plot(self.f,np.abs(self.smode[:,id])**2,'b',w,np.abs(fted)**2,'ro', markersize=0.1)
        return self.err, lstsq_err

    @staticmethod
    def get_lorenz(model):
        a = [(1j ** (len(model.a) - i - 1)) * a_i for i, a_i in enumerate(model.a)]
        a_iw = np.array(a)
        b = [(1j ** (len(model.b) - i - 1)) * a_i for i, a_i in enumerate(model.b)]
        b_iw = np.array(b)
        a_iw_c = np.conj(a_iw)
        b_iw_c = np.conj(b_iw)
        c = np.polymul(a_iw, a_iw_c)
        d = np.polymul(b_iw, b_iw_c)
        if len(c) == len(c[np.isreal(c)]) and len(d) == len(d[np.isreal(d)]):
            c = np.real(c)
            d = np.real(d)
            print('Done...')
        else:
            print('Something wrong!')
        w_0, c, a_0 = residue(c, d)
        w_p = np.abs(w_0)
        delta_k = -np.real(w_0)
        return w_p, delta_k, np.array([w_0, c, a_0])
    
    #public
    def __init__(self,src,ful,dt,tsteps):
        # src source tdcmt file, ful full tdcmt file, tsteps fdtd simulation steps
        #unpack data
        sr=unpack(src,tsteps) #unpack source
        fu=unpack(ful,tsteps) #unpack source
        _, self.nmodes = fu.shape
        # fft 
        pdss=np.fft.fft(sr,axis=0) #fft
        tmpf=np.fft.fft(fu,axis=0) #fft
        #normalize data
        pdss=np.abs(pdss)**2
        pdss=np.sum(pdss,axis=1) #pds spectrum of the source
        nrm=1./np.sqrt(pdss)
        for i in range(tmpf.shape[1]):
            mask = pdss < 1e2
            tmpf[:,i]=tmpf[:,i]*nrm
            tmpf[mask, i] = 0.0
        #compute quantities of interest
        dim=int(tsteps/2) #half frequency
        f=np.fft.fftfreq(tsteps+1,dt)/sc.c #frequencies        
        self.f=f[0:dim] #positive frequency
        pdsf=np.abs(tmpf)**2 #pds full
        dos=np.sum(pdsf,axis=1)
        self.dos=dos[0:dim] #dos
        self.pdss=pdss[0:dim] #pds of source
        self.smode=tmpf[0:dim,:] #normalized mode spectrum



#save data from fname to structured vtk
#save_vtk([dimx,dimy,dimz],'res/erg_1.bin','res/erg')                                                                  #save_vtk([dimx,dimy,dimz],'res/type_Ec.bin','res/index',np.dtype(np.int32))                                          
def save_vtk(n,fname,fnameout,dt=np.dtype(np.float64)):
    nx,ny,nz=int(n[0]),int(n[1]),int(n[2])
    #load and convert data                                                                                                                              
    data=np.fromfile(fname,dtype=dt)
    data=np.reshape(data,(nx,ny,nz))
    imageToVTK(fnameout,cellData = {fname : data})


#### complex fraction fittings
# N=2 #2*N order of poly
# M=200 #number of points
# w=np.linspace(-2.,2,M)
# fun=(1j*w)/((1j*w)**4+2.*(1j*w)+3.)
# a=frac_fit()
# a.fit0(w,fun,0*1/3.,N)
# out=a.out(w)
# plt.plot(w,np.abs(fun)**2,w,np.abs(out)**2,'r')
# plt.show()
class frac_fit(object):

    #get an odd line
    #returns (-1)n-1*w^2n-2 in column format
    def getline_odd(self,w,n):
        tmp=w.copy()
        for i in range(len(tmp)):
            tmp[i]=((-1)**(n+1))*(tmp[i]**(2*n-1))
        return tmp

    #get an even line
    #returns (-1)n-1*w^2n-2 in column format
    def getline_even(self,w,n):
        tmp=w.copy()
        for i in range(len(w)):
            tmp[i]=((-1)**(n))*(tmp[i]**(2*n))
        return tmp


    # fit complex data with (a_0+sum_n=1^N an(iw)^n)/(1+sum_n=1^N bn(iw)^n)
    # at M frequencies M=len(w)=len(data)
    # with given behavior at w=0 a_0=real(data(0))
    def fit0(self,w,data,data_0,N):
        M2=len(w) #2*number of frequency points
        #assemble linear system
        #RHS
        f=np.zeros((2*M2,1))
        self.a0=data_0
        for i in range(M2):
            f[i]=np.real(data[i])-self.a0
            f[M2+i]=np.imag(data[i])
        #B matrix
        B=np.zeros((2*M2,4*N))
        # assemble B 
        for n in np.arange(1,N+1): #1,...,N
            # assemble top
            eve=self.getline_even(w,n)
            odd=self.getline_odd(w,n)
            B[0:M2,2*n-1]=eve
            B[0:M2,2*N+2*n-1]=-np.real(data)*eve
            B[0:M2,2*N+2*n-2]=np.imag(data)*odd
            # assemble bottom
            B[M2::,2*n-2]=odd
            B[M2::,2*N+2*n-1]=-np.imag(data)*eve
            B[M2::,2*N+2*n-2]=-np.real(data)*odd
        # solve system
        self.f=f
        self.B=B
        #self.sol=np.linalg.lstsq(B,f,rcond=-1)
        # weighted lsq
        W=np.diag(f.flatten()**2)
        Bw=np.matmul(W,B)
        fw=np.matmul(W,f)
        self.sol=np.linalg.lstsq(Bw,fw,rcond=-1)
        
    # get output of the fit on frequency w
    def out(self,w):
        #construct poly of numerator/denominator
        sol=self.sol[0].flatten()
        dim=int(len(sol)/2) #number of coefficients at numerator or denominator
        a=np.zeros(dim+1) #plus 0 order term
        b=np.zeros(dim+1) #plus 0 order term
        a[0:-1]=sol[dim-1::-1]
        b[0:-1]=sol[-1:dim-1:-1]
        a[-1]=self.a0
        b[-1]=1.
        self.a=a
        self.b=b
        return np.polyval(a,1j*w)/np.polyval(b,1j*w)


# exponential fittings
# #test
# x=np.linspace(0,50,1000)
# y=1.4*np.cos(3.5*x-math.pi)*np.exp(-x/15.)+2.3*np.cos(1.8*x-math.pi/3.)*np.exp(-x/10.)
# plt.plot(x,y)
# plt.show()

# dx=x[1]-x[0]
# a=exp_fit()
# a.mtm(y,4,dx)

# x=np.linspace(0,50,300)
# out=a.model(x)
class exp_fit(object):
    # matrix pencil method on data x, fitted with p exponentials at sampling time dt=ts
    def mtm(self,x,p,ts):
        n=len(x)
        y=hankel(x[0:-p],x[-p-1::])
        
        y1=y[:,0:-1] #remove last column
        y2=y[:,1::] #remove first
        
        py1=pinv(y1) 
        l=eig(np.matmul(py1,y2))
        # get eigenvaues only
        l=l[0]
        # get alfa and w
        self.alfa=np.log(np.abs(l))/ts
        self.w=np.arctan2(np.imag(l),np.real(l))/ts
        
        # get amplitudes and phases
        z=np.zeros((n,p),dtype=np.complex128)
        for i in range(len(l)):
            tmp=l[i]**range(n)
            z[:,i]=tmp
            
        rz=np.real(z)
        iz=np.imag(z)

        rz[np.isinf(rz)]=sys.float_info.max*np.sign(rz[np.isinf(rz)])
        iz[np.isinf(iz)]=sys.float_info.max*np.sign(iz[np.isinf(iz)])
        
        z=rz+1j*iz
        h=np.linalg.lstsq(z,x,rcond=-1)
        h=h[0] #get solution
        
        self.amp=np.abs(h)
        self.theta=np.arctan2(np.imag(h),np.real(h))

    # build model on x
    def model(self,x):
        n=len(self.alfa)
        out=np.zeros(len(x),dtype=np.complex128)
        for i in range(n):
            out+=self.amp[i]*np.exp(1j*self.theta[i]+(self.alfa[i]+1j*self.w[i])*x)
        return np.real(out)

    # function that minimize
def myfun(d):
    a=unpack('res/ful.bin',50000)
    y=a[2000::,0]
    x=np.linspace(0,1.,len(y))
    y1=y[::100]
    x1=x[::100]
    b=exp_fit()
    b.mtm(y1,d,x1[2]-x1[1])
    out=b.model(x)
    #plt.plot(x,y,x,out)
    return np.linalg.norm(out-y)/np.linalg.norm(y), out

if __name__ == '__main__':
    resdir = 'res/'
    a=tdcmt(resdir+'src.bin',resdir+'ful.bin',1.17541e-12,50000)
    plt.figure(1)
    plt.clf()
    plt.plot(a.f,a.dos);plt.axis([0,5,0,14])
    plt.savefig('test.png')
