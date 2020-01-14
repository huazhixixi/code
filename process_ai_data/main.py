import numpy as np
from collections import namedtuple

from scipy.signal import correlate

from experiment_library.dsp import orthonormalize_signal, cd_compensation
from experiment_library.helpClass import Signal, FiberParam
from scipy.io import loadmat
SignalParam = namedtuple('SignalParam','baudrate,sps_in_fiber,center_wavelength')


class CoherentReciver(object):

    def __init__(self,signal_samples,signal_param_dict:SignalParam,span_length,tx_symbols_path,adc_rate):

        self.signal_samples = signal_samples
        self.signal = self.__init(signal_samples,signal_param_dict)
        self.span = FiberParam(length=span_length)
        self.tx_symbols = loadmat(tx_symbols_path)['tx_symbols']
        self.tx_symbols[1] = np.conj(self.tx_symbols[1])
        self.adc_rate = adc_rate
        self.symbol_length = int(np.floor(2 ** 18 * (20 / 80)))
        self.tx_symbols = self.tx_symbols[:,:self.symbol_length]

        self.demoded_symbol = []
        self.symbol_for_snr_calc = []
        self.snr_xpol = []
        self.snr_ypol = []

    def __init(self,signal_samples,signal_param:SignalParam)->Signal:
        signal = Signal(self.signal_samples,signal_param.sps_in_fiber,signal_param.center_wavelength,
                        signal_param.baudrate)
        return signal

    def dsp_processing(self):
        # remove mean value
        self.signal[:] = self.signal[:] - np.mean(self.signal[:],axis=1,keepdims=True)
        # IQ imbalance compensation
        self.signal[:] = orthonormalize_signal(self.signal[:])
        # normalize
        self.signal[:] = self.signal[:]/np.sqrt(np.mean(np.abs(self.signal[:])**2,axis=1,keepdims=True))
        # resample
        self.signal = self.resampy()
        # cd compensation
        self.signal = cd_compensation(self.signal,self.span,False)
        # frequecny_offset_estimation
        from experiment_library.phase import segment_find_freq
        # self.signal.samples = self.signal[:,100000:100000 + 400000]
        self.signal[:],freq_off = segment_find_freq(self.signal[:],self.signal.fs,8)

        all_samples = self.get_all_samples()
        # 对每一组做均衡
        from experiment_library.myequalize import equalizer
        from experiment_library.phase import superscalar
        from experiment_library.dsp import syncsignal_tx2rx
        for ix in range(len(all_samples)):
            all_samples[ix] = all_samples[ix]/np.sqrt(np.mean(np.abs(all_samples[ix])**2,axis=1,keepdims=True))

            symbols,*_ = equalizer(all_samples[ix],2,107,0.001,3,'lms',training_time=7,train_symbol=self.tx_symbols)

            tx_symbols_temp = syncsignal_tx2rx(symbols,self.tx_symbols)

            xpol_symbol,train_x,_ = superscalar(symbols[0],tx_symbols_temp[0],256,8,np.unique(self.tx_symbols[0]),0.02)
            ypol_symbol,train_y,_= superscalar(symbols[1],tx_symbols_temp[1],256,8,np.unique(self.tx_symbols[0]),0.02)

            xpol_symbol = np.atleast_2d(xpol_symbol)[0]
            ypol_symbol = np.atleast_2d(ypol_symbol)[0]

            xpol_noise = xpol_symbol - np.atleast_2d(train_x)[0]
            ypol_noise = ypol_symbol - np.atleast_2d(train_y)[0]

            noise_power = np.mean(np.abs(xpol_noise)**2)
            snr_xpol = 10*np.log10((1-noise_power)/noise_power)
            noise_power = np.mean(np.abs(ypol_noise)**2)

            snr_ypol = 10*np.log10((1-noise_power)/noise_power)
            self.demoded_symbol.append(np.vstack((xpol_symbol,ypol_symbol)))
            self.snr_xpol.append(snr_xpol)
            self.snr_ypol.append(snr_ypol)
            print(self.snr_xpol)
            print(self.snr_ypol)
        self.symbol_for_snr_calc = np.array((train_x,train_y))

    def get_index(self,symbol_tx,sample_rx,sps,total_number,visable=False):

        res = []
        indexes_xpol = []
        indexes_ypol = []
        for i in range(symbol_tx.shape[0]):
            symbol_tx_temp = symbol_tx[i, :]
            sample_rx_temp = sample_rx[i, :]

            res .append(correlate(sample_rx_temp[::sps], symbol_tx_temp))
            if visable:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(np.abs(np.atleast_2d(np.array(res))[i]))
                plt.show()

        res = np.array(res)
        res = np.abs(res)
        for _ in range(total_number):
            index_xpol = np.argmax(res[0])
            index_ypol = np.argmax(res[1])
            indexes_xpol.append(index_xpol)
            indexes_ypol.append(index_ypol)
            res[0,index_xpol] = 0
            res[1,index_ypol] = 0
        return indexes_xpol,indexes_ypol

    def get_all_samples(self):
        from experiment_library.dsp import syncsignal
        all_samples = []
        import matplotlib.pyplot as plt
        plt.ion()
        index_xpol,index_ypol = self.get_index(self.tx_symbols,self.signal[:],self.signal.sps,10)
        for xindex,yindex in zip(index_xpol,index_ypol):
            out_xpol = np.roll(self.signal[0], self.signal.sps * (-xindex - 1 + self.tx_symbols.shape[1]))[:self.signal.sps*self.symbol_length]
            out_ypol = np.roll(self.signal[1], self.signal.sps * (-yindex - 1 + self.tx_symbols.shape[1]))[:self.signal.sps * self.symbol_length]
            all_samples.append(np.vstack((out_xpol,out_ypol)))
        return all_samples

    def resampy(self):
        import resampy
        temp_array = resampy.resample(self.signal[:], self.signal.sps, 2, axis=1, filter='kaiser_fast')
        self.signal.samples = temp_array
        self.signal.sps = 2
        return self.signal



def demod_each_files(base_dir):
    import joblib
    import os
    for name in os.listdir(base_dir):
        # name = '5dbm_4.npz'
        samples = np.load(f'{base_dir}/{name}')['arr_0']
        xuhao = name.split('.')[0].split('_')[-1]
        # samples_new = np.zeros_like(samples)
        # samples_new[0] = samples[0]
        # samples_new[1] = samples[1]
        # samples = samples_new
        receiver = CoherentReciver(samples, SignalParam(20e9, 100e9 / 20e9, 1550e-9), 430,
                                   r'../experiment_library/txSymbols_qpsk.mat', 100e9)
        receiver.dsp_processing()
        joblib.dump(receiver, f'./demoded_data/{xuhao}')

def average(base_dir):
    xpol = []
    ypol = []
    import os
    import joblib
    for name in os.listdir(base_dir):
        symbol_x = []
        symbol_y = []
        receiver = joblib.load(base_dir + name)
        for index,(xpol_snr,ypol_snr) in enumerate(zip(receiver.snr_xpol, receiver.snr_ypol)):
            if (xpol_snr+ypol_snr)/2 >13:
                symbol_x.append(receiver.demoded_symbol[index][0])
                symbol_y.append(receiver.demoded_symbol[index][1])
        
        symbol_x = np.array(symbol_x)[:,2048:-2048]
        symbol_y = np.array(symbol_y)[:,2048:-2048]
        symbol_x = np.mean(symbol_x,axis=0)
        symbol_y = np.mean(symbol_y,axis=0)
    
        xpol.append(symbol_x)
        ypol.append(symbol_y) 
    
    xpol = np.array(xpol)
    ypol = np.array(ypol)
    xpol = np.mean(xpol,axis=0) 
    ypol = np.mean(ypol,axis=0)
    
    train_x = receiver.symbol_for_snr_calc[0]
    train_y = receiver.symbol_for_snr_calc[1]

    xpol = xpol/np.sqrt(np.mean(np.abs(xpol)**2))
    ypol = ypol/np.sqrt(np.mean(np.abs(ypol)**2))
    xpol_noise = xpol - np.atleast_2d(train_x)[0,2048:-2048]
    ypol_noise = ypol - np.atleast_2d(train_y)[0,2048:-2048]

    noise_power = np.mean(np.abs(xpol_noise) ** 2)
    snr_xpol = 10 * np.log10((1 - noise_power) / noise_power)
    noise_power = np.mean(np.abs(ypol_noise) ** 2)

    snr_ypol = 10 * np.log10((1 - noise_power) / noise_power)

    print(snr_xpol,snr_ypol)
    
if __name__ == '__main__':
    #demod_each_files('npzdata')
    average('demoded_data/')
