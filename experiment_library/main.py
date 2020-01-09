# from tools import read_data
import numpy as np

from dsp_tool import normalise_and_center
from helpClass import FiberParam
from helpClass import Signal
from dsp import cd_compensation, syncsignal, syncsignal_tx2rx
from dsp import orthonormalize_signal
from phase import superscalar
from dsp import resampling
from dsp import remove_dc
from myequalize import equalizer
from phase import segment_find_freq

class CoherentReceiver(object):
    def __init__(self,samples,dac_rate,adc_rate,symbol_rate,total_length,tx_symbol,unit):
        self.samples = samples
        self.dac_rate = dac_rate
        self.adc_rate = adc_rate
        self.symbol_rate = symbol_rate
        self.signal = Signal(samples,sps=dac_rate/symbol_rate,center_wavelength=1550e-9,symbol_rate=symbol_rate*{'hz':1,'ghz':1e9}[unit])
        self.prop_length = total_length
        self.fiber = FiberParam(length=total_length)
        self.tx_symbol = tx_symbol

    def dsp_processing(self):

        data_len = int(np.ceil(2 ** 18 * (self.symbol_rate / self.dac_rate)))
        self.signal = remove_dc(self.signal)
        self.signal = resampling(self.signal,self.adc_rate,self.symbol_rate,new_sps=2)
        self.signal = orthonormalize_signal(self.signal,self.signal.sps)
        # self.signal,freq_offset = segment_find_freq(self.signal,self.signal.fs,group=8)
        self.tx_symbol = self.tx_symbol[:,:data_len]
        self.signal.samples = self.signal[:,10000:10000+400000]
        self.signal = cd_compensation(self.signal,self.fiber)

        self.signal,freq_offset = segment_find_freq(self.signal,self.signal.fs,group=8)

        self.signal = syncsignal(self.tx_symbol,self.signal,2,visable=False)
        self.signal.samples = self.signal.samples[:,:data_len * self.signal.sps]
        self.signal = normalise_and_center(self.signal)
        self.signal.samples,*_ = equalizer(self.signal[:],self.signal.sps,ntaps=67,mu=0.0004,iter_number=13,method='lms',training_time=13,train_symbol=self.tx_symbol)
        import matplotlib.pyplot as plt

        # plt.scatter(self.signal.samples[0].real,self.signal.samples[0].imag,c='b')
        #
        # plt.scatter(self.signal.samples[1].real,self.signal.samples[1].imag,c='b')
        self.tx_symbol = syncsignal_tx2rx(self.signal[:],self.tx_symbol)
        self.tx_symbol = self.tx_symbol[:,:self.signal.shape[1]]
        self.signal = normalise_and_center(self.signal)

        xpol = self.signal[0]
        ypol = self.signal[1]
        xpol, xpol_train_symbol = superscalar(xpol,self.tx_symbol[0],block_length=256,pilot_number=16,constl=np.unique(self.tx_symbol[0]),g=0.01,filter_n=20)
        ypol, ypol_train_symbol = superscalar(ypol, self.tx_symbol[1], block_length=256, pilot_number=16,
                                              constl=np.unique(self.tx_symbol[0]), g=0.01, filter_n=20)
        xpol = np.atleast_2d(xpol)
        ypol = np.atleast_2d(ypol)
        plt.scatter(xpol[0].real, xpol[0].imag, c='b')

        plt.scatter(ypol[0].real, ypol[0].imag, c='b')
        self.snr = self.calc_snr(xpol,xpol_train_symbol,ypol,ypol_train_symbol)
        print(self.snr)
    def calc_snr(self,xpol,xpol_train,ypol,ypol_train):
        noisex = xpol - xpol_train
        power = np.mean(np.abs(noisex)**2)
        snrx = 10*np.log10((1-power)/power)

        noisey = ypol - ypol_train
        power = np.mean(np.abs(noisey) ** 2)
        snry = 10 * np.log10((1 - power) / power)
        return snrx,snry

def main():
    receivers = []
    for name in range(20):
        signal_samples = np.load(f'./5dbmdata/{name}.npz')['arr_0']
        xpol = signal_samples[0]
        ypol = signal_samples[1]
        signal_samples = np.array((xpol,ypol))
        from scipy.io import loadmat
        symbols = loadmat('txSymbols_qpsk.mat')['tx_symbols']
        symbols[1] = np.conj(symbols[1])
        receiver = CoherentReceiver(signal_samples,80,100,20,432,symbols,unit='ghz')
        receiver.dsp_processing()
        receivers.append(receiver)


main()