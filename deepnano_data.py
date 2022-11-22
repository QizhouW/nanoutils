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
import time
import pandas as pd
from phase_utils import parse_phase,diff_phase,extract_wl
from skimage import io
from utils import mkdir
import h5py



def readey(resdir, file):
    path = os.path.join(resdir, file)
    data = np.fromfile(path, dtype=np.float64)
    return data


def merge_csv(list_of_data, outputfile):
    tmp = pd.concat(list_of_data).drop_duplicates(keep=False)
    tmp.reset_index(drop=True)
    
    tmp.to_csv(outputfile)
    print('Merged data length: ', len(tmp))
    return True


def clear_csv(all_data, parsed_data, outputfile):
    tmp = pd.concat([all_data, parsed_data, parsed_data]).drop_duplicates(keep=False)
    tmp.to_csv(outputfile)
    print('Clean data length: ', len(tmp))
    return True


def raw_to_hdf5(raw_dir,name,compression=4):
    ## Before delete, check if the parse is complete, use this externally, no need to integrate
    output_dir = os.path.join(raw_dir, name, 'hdf5')
    mkdir(output_dir)
    len_of_data=[]
    if  not os.path.isdir(os.path.join(raw_dir, name,'resdata')):
        print(f'Raw data does not exist, might be processed in {name}')
        return -1

    d=os.path.join(raw_dir, name)
    dataitem=pd.read_csv(os.path.join(d,  f'data_{name}.csv'), index_col=0)
    len_of_data.append(len(dataitem))
    print(f'Processing {name} length {len(dataitem)}')

    with h5py.File(os.path.join(output_dir,"data.hdf5"), "w") as f:
        gex = f.create_group("ex")
        gey = f.create_group("ey")
        gshape= f.create_group("geo")
        for idx,item in dataitem.iterrows():
            try:
                ex = readey(os.path.join(d,'resdata'), f'ex_{item.prefix}.bin')
                ex = ex.reshape(-1, 4)
                gex.create_dataset(item.prefix,ex.shape,dtype='float64',compression="gzip", compression_opts=compression)
                gex[item.prefix][...] = ex

                ey = readey(os.path.join(d,'resdata'), f'ey_{item.prefix}.bin')
                ey = ey.reshape(-1, 4)
                gey.create_dataset(item.prefix,ey.shape,dtype='float64',compression="gzip", compression_opts=compression)
                gey[item.prefix][...] = ey

                geo = io.imread(os.path.join(d,'geo',f'{item.prefix}.png'))
                geo = geo[:, :, 0]
                geo[geo == 8] = 0
                geo[geo == 255] = 1

                gshape.create_dataset(item.prefix,geo.shape,dtype='uint8',compression="gzip", compression_opts=compression)
                gshape[item.prefix][...] = geo
                
            except FileNotFoundError:
                print(f'Processing file {idx+1}. ignoring incomplete simulations.')
    return 0 



def merge_parsed(parsed_dir,output_dir,dim):
    output_dir=os.path.join(output_dir,str(dim))
    item_ls = os.listdir(parsed_dir)
    mkdir(output_dir,rm=True)
    output_datafile = os.path.join(output_dir, 'data.csv')
    attribute_ls = ['amp', 'phase', 'err', 'label','geo']
    with open(os.path.join(output_dir, 'Merge.log'), 'a') as f:
        timestr = time.strftime("%Y%m%d_%H%M", time.localtime())
        f.write('Data processing log' + timestr)
    dir_ls = []
    list_of_data=[]
    for item in item_ls:
        with open(os.path.join(output_dir, 'Merge.log'), 'a') as f:
            if os.path.isdir(os.path.join(parsed_dir, item)):
                d=os.path.join(parsed_dir, item)
                dataitem=pd.read_csv(os.path.join(d, 'data.csv'), index_col=0)
                if np.unique(dataitem.nx)==np.unique(dim):
                    list_of_data.append(dataitem)
                    f.write(f'Added data from {item}\n')
                    dir_ls.append(d)
    print(dir_ls)
    merge_csv(list_of_data, output_datafile)
    for att in attribute_ls:
        merged_att = []
        for d in dir_ls:
            merged_att.append(np.load(os.path.join(d, f'{att}.npy'), allow_pickle=True))
        merged_att = np.concatenate(merged_att)
        np.save(os.path.join(output_dir, f'{att}.npy'), merged_att)
    return True


class SingleSample():
    def __init__(self,dir,samplename):
        ex = readey(dir, f'ex_{samplename}.bin')
        srcex = ex.reshape(-1, 4)[:, 1]
        msex, psrcex, asrcex = parse_phase(srcex, m=20, wl1=0.38, wl2=0.7, D=100, dt=4.81458e-12, skip=10000)
        ey = readey(dir, f'ey_{samplename}.bin')
        srcey = ey.reshape(-1, 4)[:, 2]
        msey, psrcey, asrcey = parse_phase(srcey, m=20, wl1=0.38, wl2=0.7, D=100, dt=4.81458e-12, skip=10000)
        self.ex={'phase':psrcex,'amp':asrcex,'mse':msex}
        self.ey={'phase':psrcey,'amp':asrcey,'mse':msey}
        return

    def __repr__(self):
        return 'ex:\n'+str(self.ex)+'\ney:\n'+str(self.ey)

class Dataset():
    def __init__(self, src_dir, ful_dir, csv_file, output_dir):
        self.src_dir = src_dir
        self.ful_dir=ful_dir
        self.output_dir=output_dir
        self.csv_file=csv_file
        self.data_log=pd.read_csv(csv_file, index_col=0)
        print(f'Loaded total {len(self.data_log)} samples')
        self.unit_sizes = np.unique(self.data_log.nx)
        self.parse_src_data()

    def parse_src_data(self):
        self.src_data={}
        for Nx in self.unit_sizes:
            self.src_data[Nx]=SingleSample(self.src_dir,f'src{Nx}')
        return
    
    def parse_data(self,threshold=0.01,ends=None):
        self.dphase=[]
        self.transmission=[]
        self.geometry=[]
        self.labels=[]
        self.high_err=[]
        self.csvtosave=[]
        self.error=[]
        for idx, item in enumerate(self.data_log.iterrows()):
            try:
                if idx%1000==0:
                    print(idx)
                item = item[1]
                Nx=item.nx
                src_data=self.src_data[Nx]
                ful_data=SingleSample(self.ful_dir+'/resdata',item.prefix)
                dphasex=diff_phase(src_data.ex['phase'],ful_data.ex['phase'],unwrap=False)
                dphasey=diff_phase(src_data.ey['phase'],ful_data.ey['phase'],unwrap=False)
                self.dphase.append(([dphasex,dphasey]))
                self.transmission.append([ful_data.ex['amp']/src_data.ex['amp'],ful_data.ey['amp']/src_data.ey['amp']])
                self.error.append([ful_data.ex['mse'], ful_data.ey['mse']])
                img = io.imread(self.ful_dir + f'/geo/{item.prefix}.png')
                img = img[:, :, 0]
                img[img == 8] = 0
                img[img == 255] = 1
                # assert np.unique(img)==[0,1]
                self.geometry.append(img)
                label=np.load(self.ful_dir + f'/geo/{item.prefix}.npy',allow_pickle=True)
                self.labels.append(label)
                if ful_data.ex['mse'] > threshold or ful_data.ey['mse'] > threshold:
                    self.high_err.append(idx)
                else:
                    self.csvtosave.append(item)
            except:
                print(f'Error in parsing: {idx}')
                self.high_err.append(idx)
            if ends:
                if idx == ends-1:
                    break
        print('High error data: ', len(self.high_err))
        print('Normal data: ', len(self.csvtosave))

    def save(self,name):
        sample_data=readey(self.src_dir,f'ex_src100.bin')
        wl_list = extract_wl(sample_data, wl1=0.38, wl2=0.7, D=100, m=20, dt=4.81458e-12, skip=10000)
        geos = np.array(self.geometry)
        errorls = np.array(self.error)
        phasels = np.array(self.dphase)
        ampls = np.array(self.transmission)
        labels=np.array(self.labels)
        geos = np.delete(geos, self.high_err, 0)
        phasels = np.delete(phasels, self.high_err, 0)
        ampls = np.delete(ampls, self.high_err, 0)
        errorls = np.delete(errorls, self.high_err, 0)
        labels=np.delete(labels, self.high_err, 0)
        mkdir(f'./{self.output_dir}/{name}')
        np.save(f'./{self.output_dir}/{name}/geo.npy', geos)
        csvtosave = pd.DataFrame(self.csvtosave)
        csvtosave.to_csv(f'./{self.output_dir}/{name}/data.csv')
        np.save(f'./{self.output_dir}/{name}/err.npy', errorls)
        np.save(f'./{self.output_dir}/{name}/phase.npy', phasels)
        np.save(f'./{self.output_dir}/{name}/amp.npy', ampls)
        np.save(f'./{self.output_dir}/{name}/label.npy', labels)
        wl_list = np.round(wl_list, 3)
        np.savetxt(f'./{self.output_dir}/{name}/wavelengths.txt', wl_list)

