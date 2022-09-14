from scipy.interpolate import interp1d
import numpy as np
import math  
import os

#------------------------ MATERIAL

rep='https://www.filmetrics.com/refractive-index-database/download/' #repository of indices

#wavelengths in nanometers

class material(object):

    ##### Init fname is name of material
    def __init__(self, file, source='web', **kargs):
        """file - [fil]"""
        self.file = file
        if file is np.array:
            source = 'array'
        self.load_material(source,**kargs)

    def load_material(self, source, **kargs):
        if source == 'web':
            self.load_from_web(**kargs)
        elif source == 'file':
            self.air = False
            self.load_from_file(**kargs)
        elif source == 'array':
            self.load_from_array()
        else:
            FileNotFoundError('Unrecognized Source!')

    def load_from_web(self, **kargs):
        fname = self.file
        if (fname != 'AIR'):
            self.air = False
            # load file
            if (not os.path.isfile(fname)):
                # changes spaces
                dname = ''
                for i in fname:
                    if (i != ' '):
                        dname += i
                    else:
                        dname += '%20'
                cmd = 'wget --no-check-certificate ' + rep + dname
                print(cmd)
                os.system(cmd)
            # load data
            a = open(fname, 'r')
            b = a.read()
            b = '#' + b  # add a comment
            a.close()
            a = open(fname, 'w')
            a.write(b)
            a.close()
            data = np.loadtxt(fname, comments='#')
            self.wl = data[:, 0]  # in nm
            self.n = data[:, 1]
            self.k = data[:, 2]
            eps = (self.n + 1j * self.k) ** 2
            self.er = np.real(eps)
            self.ei = np.imag(eps)

            self.ni = interp1d(self.wl, self.n)
            self.ki = interp1d(self.wl, self.k)
            self.eri = interp1d(self.wl, self.er)
            self.eii = interp1d(self.wl, self.ei)
        else:
            self.air=True

    def load_from_file(self, **kargs):
        """
        method to load dispersion from txt file
        :param kargs: key arguments which transfered to loadtxt function. Specify skiprows argument if you have
        commentaries in your data file/
        :return: None
        """
        fname = self.file
        data = np.loadtxt(fname, **kargs)
        self.wl = data[:, 0]  # in nm
        self.n = data[:, 1]
        self.k = data[:, 2]
        eps = (self.n + 1j * self.k) ** 2
        self.er = np.real(eps)
        self.ei = np.imag(eps)

        self.ni = interp1d(self.wl, self.n)
        self.ki = interp1d(self.wl, self.k)
        self.eri = interp1d(self.wl, self.er)
        self.eii = interp1d(self.wl, self.ei)

    def load_from_array(self):
        data = self.file
        self.wl = data[:, 0]  # in nm
        self.n = data[:, 1]
        self.k = data[:, 2]
        eps = (self.n + 1j * self.k) ** 2
        self.er = np.real(eps)
        self.ei = np.imag(eps)

        self.ni = interp1d(self.wl, self.n)
        self.ki = interp1d(self.wl, self.k)
        self.eri = interp1d(self.wl, self.er)
        self.eii = interp1d(self.wl, self.ei)

    # get n and k at specific wavelengths wl in nm
    def get_nk(self,wl):
        #interp
        if(self.air):            
            return np.ones(len(wl)),np.zeros(len(wl))
        else:
            n=self.ni(wl)
            k=self.ki(wl)
            return n,k

    # get n and k at specific wavelengths wl in nm
    def get_eri(self,wl):
        #interp
        if(self.air):
            return np.ones(len(wl)),np.zeros(len(wl))
        else:
            er=self.eri(wl)
            ei=self.eii(wl)
            return er,ei
