import numpy as np
import pandas as pd
from dsp import remove_dc, syncsignal, syncsignal_tx2rx

from scipy.io import loadmat

from dsp import resampling
from dsp_tool import normalise_and_center
from phase import superscalar
from phase import segment_find_freq
from helpClass import Signal,FiberParam
import matplotlib.pyplot as plt
import os

def scatterplot(samples):
    plt.figure()
    samples = np.atleast_2d(samples)
    plt.scatter(samples[0].real,samples[0].imag,color='b',s=1)
    plt.show(block=False)




def read_data():
    import tqdm
    BASE_DIR = './20191229'
    import os
    allpowerdir =os.listdir(BASE_DIR)


    for powerdir in tqdm.tqdm(allpowerdir):
        for groupdir in os.listdir(BASE_DIR+'/'+powerdir):
            for file_name in os.listdir(BASE_DIR+'/'+powerdir+'/'+groupdir):
                if 'Ch1' in file_name:
                    ch1_dir = BASE_DIR+'/'+powerdir+'/'+groupdir+'/'+file_name
                if 'Ch2' in file_name:
                    ch2_dir = BASE_DIR+'/'+powerdir+'/'+groupdir+ '/' + file_name
                if 'Ch3' in file_name:
                    ch3_dir = BASE_DIR+'/'+powerdir+'/'+groupdir+'/' + file_name
                if 'Ch4' in file_name:
                    ch4_dir = BASE_DIR+'/'+powerdir+'/'+groupdir+ '/' + file_name

            power = powerdir[0]

            ith = groupdir

            ch1 = pd.read_csv(ch1_dir, header=None)
            ch1 = ch1.iloc[:, 4].values

            ch2 = pd.read_csv(ch2_dir,  header=None)
            ch2 = ch2.iloc[:, 4].values

            ch3 = pd.read_csv(ch3_dir, header=None)
            ch3 = ch3.iloc[:, 4].values

            ch4 = pd.read_csv(ch4_dir, header=None)
            ch4 = ch4.iloc[:, 4].values

            ch1.shape = 1, -1
            ch2.shape = 1, -1
            ch3.shape = 1, -1
            ch4.shape = 1, -1
            ch2 = np.roll(ch2, -1, axis=1)
            ch4 = np.roll(ch4, 2, axis=1)

            xpol = ch1 + 1j * ch2
            ypol = ch3 + 1j * ch4

            np.savez('./npzdata_new/'+power+'dbm_'+ith+'ith',np.vstack((xpol,ypol)))


from dsp import orthonormalize_signal
from dsp import cd_compensation
from scipy.constants import c
from scipy.signal import welch
def main():
    import pathlib
    BASE_DIR = pathlib.Path('npzdata_new')
    res = {}
    file_names = os.listdir('npzdata_new')
    file_names = sorted(file_names,key=lambda name:int(name[0]))
    for file in file_names:
        print(file)
        file = '5dbm_1ith.npz'
        samples =np.load(BASE_DIR/file)['arr_0']
        # samples = loadmat('to_do.mat')
        # xpol = samples['recv_samples']
        # ypol = samples['recv_samples2']
        xpol = samples[1]
        ypol = samples[0]
        data_len = int(np.ceil(2 ** 17 * (30 / 80)))
        # tx_symbols = loadmat('txSymbols_qpsk.mat')['tx_symbols'][:, :data_len]
        # tx_symbols[1] = np.conj(tx_symbols[1])
        # tx_symbols = tx_symbols[:, :data_len]
        xpol = remove_dc(xpol)
        ypol = remove_dc(ypol)
        xpol = resampling(xpol, 100, 30)
        ypol = resampling(ypol, 100, 30)
        xpol = orthonormalize_signal(xpol, os=1)
        ypol = orthonormalize_signal(ypol, os=1)

        signal = Signal(np.vstack((xpol, ypol)),2,c/193.1e12,30e9)
        span = FiberParam(length=430)
        signal = cd_compensation(signal,span)
        signal.samples= signal.samples[:, 300000:300000 + 400000]
        signal.samples, freq_offset = segment_find_freq(signal,signal.fs,8)
        signal.samples = normalise_and_center(signal[:])
        # signal.samples = syncsignal(tx_symbols, signal[:], 2, True)
        # signal.samples = signal.samples[:,:data_len*signal.sps]
        from myequalize import equalizer
        signal.samples,*others= equalizer(signal[:],signal.sps,31,1e-4,17,method='cma')
        signal[:] = syncsignal(tx_symbols, signal[:], 1, 0)
        # signal.sps = 1
        # tx_symbols = syncsignal_tx2rx(signal[:],tx_symbols)
        # tx_symbols = tx_symbols[:,:signal.shape[1]]
        signal[:] = normalise_and_center(signal[:])

        signal.samples = signal.samples[:,:data_len]
        xpol, tx_xpol = superscalar(signal[0], tx_symbols[0], 256, 4, constl=np.unique(tx_symbols[0]), g=0.02)
        ypol, tx_ypol = superscalar(signal[1], tx_symbols[1], 256, 4, constl=np.unique(tx_symbols[1]), g=0.02)

        xpol = np.atleast_2d(xpol)
        ypol = np.atleast_2d(ypol)
        scatterplot(xpol[0,2048:-1-2048])
        # scatterplot(ypol[0,2048:-1-2048])

        noise = xpol[0,2048:-1-2048] - tx_xpol[0,2048:-1-2048]

        power = np.mean(np.abs(xpol) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)

        snrx = (power - noise_power) / noise_power
        print(10 * np.log10(snrx))
        noise = ypol[0,2048:-1-2048] - tx_ypol[0,2048:-1-2048]

        power = np.mean(np.abs(xpol) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)

        snry = (power - noise_power) / noise_power
        print(10 * np.log10(snry))


        dbm = file[0]
        res.setdefault(dbm,[])
        res[dbm].append([snrx,snry])



    return res

if __name__ == '__main__':

    main()