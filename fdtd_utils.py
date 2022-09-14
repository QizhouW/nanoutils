import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import scipy.io as sio
import math
from scipy.linalg import hankel
from scipy.linalg import pinv
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.optimize import lsq_linear
#%matplotlib
import scipy.constants as sc
import time
import os
import math
import sys
import materials as mt
from colors import color
import warnings
warnings.filterwarnings('ignore')

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
def unpack(file,tsteps, skip=0):
    data=np.fromfile(file,dtype=np.float64)
    nmodes=int(len(data)/(tsteps+1))
    data=data.reshape(tsteps+1,nmodes)
    return data[skip:,:]


####### TDCMT CLASS
class tdcmt(object):

    # private
    def fit_best(self, mode, err_th=1, fmax=10, max_N=3):
        thresholds = np.arange(0, 0.5, 0.1)
        lstsq_err = 1e10
        for N in range(1, max_N + 1):
            for th in thresholds:
                fit_err, lstsq_err = self.fitone(mode, th, fmax, N, plot=False)
                if lstsq_err <= err_th:
                    return self.a, lstsq_err

        return None, lstsq_err

    def fit_given_modes(self, modes_idxs, rewrite=True, *pargs, **kargs):
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
            self.models[i], errs[i] = self.fit_best(i, *pargs, **kargs)

        fails = np.where(self.models == None)
        return errs, fails[0]

    def get_lstsq_error(self, w, fun, fit):
        return np.sum(np.abs(fit - fun) ** 2)

    # fit the spectrum of mode id from th*max to max below fmax with minimum number of poles N
    def fitone(self, id, th, fmax, N=3, plot=True):
        ff = self.f <= fmax
        max = np.max(np.abs(self.smode[ff, id]))
        msk = (np.abs(self.smode[:, id]) >= th * max) * (self.f <= fmax)  # (self.f>=f0)*(self.f<=f1)
        self.msk = msk
        w = self.f[msk]
        fun = self.smode[msk, id]
        a = frac_fit()
        a.fit0(w, fun, np.real(fun[0]), N)
        fted = a.out(w)
        self.err = a.sol[1]
        self.a = a
        lstsq_err = self.get_lstsq_error(ff, self.smode[ff, id], a.out(self.f[ff]))
        if plot and lstsq_err < 1e2:
            print(self.err, lstsq_err, id)
            plt.figure(1)
            # plt.clf()
            plt.plot(self.f, np.abs(self.smode[:, id]) ** 2, 'b', w, np.abs(fted) ** 2, 'ro', markersize=0.1)
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

    # public
    def __init__(self, src, ful, dt, tsteps):
        """
        :param src: path to source file
        :param ful: path to ful file
        :param dt: dt (from NANOCPP output)
        :param tsteps: time steps (from NANOCPP output)
        """
        # src source tdcmt file, ful full tdcmt file, tsteps fdtd simulation steps
        # unpack data
        sr = unpack(src, tsteps)  # unpack source
        fu = unpack(ful, tsteps)  # unpack source
        _, self.nmodes = fu.shape
        # fft
        pdss = np.fft.fft(sr, axis=0)  # fft
        tmpf = np.fft.fft(fu, axis=0)  # fft
        # normalize data
        pdss = np.abs(pdss) ** 2
        pdss = np.sum(pdss, axis=1)  # pds spectrum of the source
        nrm = 1. / np.sqrt(pdss)
        for i in range(tmpf.shape[1]):
            mask = pdss < pdss.max()*1e-3
            tmpf[:, i] = tmpf[:, i] * nrm
            tmpf[mask, i] = 0.0
        # compute quantities of interest
        dim = int(tsteps / 2)  # half frequency
        f = np.fft.fftfreq(tsteps + 1, dt) / sc.c  # frequencies
        self.f = f[0:dim]  # positive frequency
        pdsf = np.abs(tmpf) ** 2  # pds full
        dos = np.sum(pdsf, axis=1)
        self.dos = dos[0:dim]  # dos
        self.pdss = pdss[0:dim]  # pds of source
        self.smode = tmpf[0:dim, :]  # normalized mode spectrum


class harm():
    def unpack(self,file,tsteps,skip=0):
        # hx, ex, ey, hy,
        stride = 2
        if self.dim == 3:
            stride = 4
        data = np.fromfile(file, dtype=np.float64)
        nmodes = int(len(data) /stride /(tsteps + 1))
        data = data.reshape(tsteps+1, stride, nmodes)
        return data[skip:]

    def __init__(self, src, ful, dt, tsteps, dim = 3, skip=0):
        """
        :param src: path to source file
        :param ful: path to ful file
        :param dt: dt (from NANOCPP output)
        :param tsteps: time steps (from NANOCPP output)
        :param dim: dimension
        :param skip: skip few timesteps (for CW source)
        """
        self.dim = dim
        tsteps -= skip
        dim = int(tsteps / 2)  # half frequency
        sr = self.unpack(src, tsteps, skip)
        fu = self.unpack(ful, tsteps, skip)
        sr = np.expand_dims(sr[:,:,0], axis=2)
        f = np.fft.fftfreq(tsteps + 1, dt) / sc.c  # frequencies
        self.f = f[1:dim]
        self.nmodes = fu.shape[2]
        pdss = np.fft.fft(sr, axis=0)[1:dim]  # fft
        pdsf = np.fft.fft(fu, axis=0)[1:dim]  # fft
        poy_sr = self.get_poy(pdss)
        self.poy_fu = self.get_poy(pdsf)
        msk = (poy_sr.squeeze() > poy_sr.squeeze().max() * 1e-2)
        self.wl = 1 / self.f[msk]
        self.poy_sr = poy_sr
        self.transm = self.poy_fu / poy_sr
        self.transm = self.transm[msk]
        #self.get_color()

    def get_poy(self, x):
        if self.dim == 3:
            return np.real(x[:,1] * x[:,3].conj() - x[:,0]*x[:,2].conj())
        else:
            return np.real(x[:,0]*x[:,1].conj())

    def get_color(self, units=1e-6, plot=False, mode=0):
        a = np.stack([self.wl,self.transm[:,mode]],axis=1)
        self.color = color(a, units = units)
        if plot:
            self.color.plot_rgb()
            self.color.plot_cie()
        return self.color.xy



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

    #get an odd line fpr fit1
    #returns (-1)n*w^2n+1 in column format
    def getline_odd1(self,w,n):
        tmp=w.copy()
        for i in range(len(tmp)):
            tmp[i]=((-1)**(n))*(tmp[i]**(2*n+1))
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
        a0=data_0
        for i in range(M2):
            f[i]=np.real(data[i])-a0
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
        self.sol=np.linalg.lstsq(B,f,rcond=-1)
        #         # weighted lsq
        W=np.diag(f.flatten()**2)
        Bw=np.matmul(W,B)
        fw=np.matmul(W,f)
        self.sol=np.linalg.lstsq(Bw,fw,rcond=-1)
        # calculate a and b coefficients
        sol=self.sol[0].flatten()
        dim=int(len(sol)/2) #number of coefficients at numerator or denominator
        a=np.zeros(dim+1) #plus 0 order term
        b=np.zeros(dim+1) #plus 0 order term
        a[0:-1]=sol[dim-1::-1]
        b[0:-1]=sol[-1:dim-1:-1]
        a[-1]=a0
        b[-1]=1.
        self.a=a
        self.b=b


    # fit complex data with (sum_n=0^N an(iw)^n)/(sum_n=0^N bn(iw)^n)
    # at M frequencies M=len(w)=len(data)
    # with behavior at w=infty: real part positive=ainf, imaginary part=-0
    # with behavior at zero: real part=positive, imaginary part=-0
    # test fit1
    #fit1
    # get experimental data
#     import materials as mt
#     M=100#2*N #number of points=2*N
#     wl=np.linspace(300,1000,M)
#     w=1./wl #frequencies in unit of 2pi/c (nm^-1)
#     b=mt.material('Si')
#     tmp=b.get_eri(wl)
#     fun=tmp[0]-1j*tmp[1]
#     # do fitting
#     N=2 #N order of poly
#     a=frac_fit()
#     a.fit1(w,fun,N,1.)

#     # check critical frequency
#     dx=1000./200
#     a.check_FDTD(dx)

#     # get fdtd string and save it to file
#     a.get_FDTD_string()
#     wn=2.*math.pi*sc.c/wl*1e9
#     out=np.polyval(a.ar,1j*wn)/np.polyval(a.br,1j*wn)
#     plt.figure(1)
#     plt.clf()
#     plt.plot(wl,out.real,'b',wl,out.imag,'r',wl,tmp[0],'o',wl,-tmp[1],'o')

#     plt.show()
    def fit1(self,w,data,N,ainf=1.):
        #assemble linear system
        #RHS
        M=len(w)
        f=np.zeros((2*M,1))
        a2n=1.1
        b2n=1.
        for i in range(M):
            tmp=((-1)**N)*(w[i]**(2*N))
            tmp1=tmp*w[i]
            f[i]=-a2n*tmp+np.real(data[i])*b2n*tmp-np.imag(data[i])*tmp1
            f[M+i]=-ainf*tmp1+np.real(data[i])*tmp1+np.imag(data[i])*b2n*tmp
        #B matrix
        B=np.zeros((2*M,4*N))
        # assemble B 
        for n in np.arange(N): #0,...,N-1
            # assemble top
            eve=self.getline_even(w,n)
            odd=self.getline_odd1(w,n)
            B[0:M,2*n]=eve
            B[0:M,2*N+2*n]=-np.real(data)*eve
            B[0:M,2*N+2*n+1]=np.imag(data)*odd
            # assemble bottom
            B[M::,2*n+1]=odd
            B[M::,2*N+2*n]=-np.imag(data)*eve
            B[M::,2*N+2*n+1]=-np.real(data)*odd
        # solve system        
        self.f=f
        self.B=B
#        self.sol=np.linalg.lstsq(B,f,rcond=-1)
#        sol=self.sol[0].flatten()
        self.sol=lsq_linear(B,f.flatten(),bounds=(0.,np.inf))
        sol=self.sol.x
        dim=int(len(sol)/2) #number of coefficients at numerator or denominator
        a=np.zeros(dim+2) #plus 2 order term 2n,2n+1
        b=np.zeros(dim+2) #plus 2 order term
        a[0]=ainf
        b[0]=1.
        a[1]=a2n
        b[1]=b2n
        a[2::]=sol[dim-1::-1]
        b[2::]=sol[-1:dim-1:-1]
        #slightly correct behavior at 0
        a0=a[-1]
        b0=b[-1]
        a1=a[-2]
        b1=b[-2]
        if(a0<a1/b1*b0):
            #correct behavior at 0
            alp=a1/b1*b0/a0*(1.+np.finfo(float).eps)
            a[-1]=a[-1]*alp
        self.a=a
        self.b=b
        
    # get output of the fit on frequency w
    def out(self,w):
        return np.polyval(self.a,1j*w)/np.polyval(self.b,1j*w)

    # check behavior at critical frequency for FDTD with spatial resolution dx=dy=dz and CFL
    def check_FDTD(self,dx,cfl=2,units=1e-9):
        a=units
        wcr=cfl*math.sqrt(2)/math.pi/dx
        tmp=self.out(wcr*3./4.)
        self.wcr = wcr
        print('critical w (2pi/c) = %e, epsilon(3w/4) = (%e,%e)' % (wcr,tmp.real,tmp.imag))

    # get dispersive parameters formatted for nanocpp and save it in file fname
    def get_FDTD_string(self,fname='str.txt',units=1e-9):
        #get number of poles
        np=len(self.a)-1
        #rescale coefficients
        a=2.*math.pi*sc.c/units
        self.ar=self.a.copy()
        self.br=self.b.copy()
        for i in range(np+1):
            tmp=a**(i)
            self.ar[i]*=tmp
            self.br[i]*=tmp
        ar = self.ar[::-1]
        br = self.br[::-1]
        str='%i' % np
        for i in range(np+1):
            str+=" %.3e" % ar[i]
        for i in range(np+1):
            str+=" %.3e" % br[i]
        if fname:
            f=open(fname,"w")
            f.write(str)
            f.close()
        return str
                
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
    a=unpack('ht.bin',10000)
    y=a[2000::,0]
    x=np.linspace(0,1.,len(y))
    y1=y[::100]
    x1=x[::100]
    b=exp_fit()
    b.mtm(y1,d,x1[2]-x1[1])
    out=b.model(x)
    #plt.plot(x,y,x,out)
    return np.linalg.norm(out-y)/np.linalg.norm(y)


class disp_fit(frac_fit):
    def __init__(self, material):
        frac_fit.__init__(self)
        self.mt = material
    def fit(self, N=1, wl=None, units = 1e-9, dx=5):
        if wl is None:
            wl = np.linspace(300,1000,100)
        w=1./wl #frequencies in unit of 2pi/c (nm^-1)
        b=self.mt
        tmp=b.get_eri(wl)
        fun=tmp[0]-1j*tmp[1]
        #do fitting
        self.fit1(w,fun,N,1.)

        #check critical frequency
        self.check_FDTD(dx,units=units)

        #get fdtd string and save it to file
        self.get_FDTD_string(units=units)
        wn=2.*math.pi*sc.c/wl/units
        self.wn = wn
        #wn = np.linspace(1e1, 5e14, 100)
        out=np.polyval(self.ar,1j*wn)/np.polyval(self.br,1j*wn)
        plt.figure(1)
        plt.clf()
        plt.plot(wn,out.real,'b',wn,out.imag,'r',wn,tmp[0],'o',wn,-tmp[1],'o')
        #plt.plot(wl,out.real, 'b', wl, out.imag, 'r')
        plt.show()

    def plot(self, w):
        out = np.polyval(self.a, 1j * w) / np.polyval(self.b, 1j * w)
        plt.plot((w-self.wcr)/self.wcr, out.real, 'b', (w-self.wcr)/self.wcr, out.imag, 'r')
        plt.show()

#fit1
#get experimental data
if __name__ == "__main__":
    pass

# a=tdcmt('td_src.bin','td_ful.bin',5.89664e-12,50000)
# plt.figure(1)
# plt.clf()
# plt.plot(a.f,a.dos);plt.axis([0,5,0,14])
