import numpy as np
from collections import namedtuple

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

    def get_all_samples(self):
        from experiment_library.dsp import syncsignal
        self.signal[:] = syncsignal(self.tx_symbols,self.signal[:],self.signal.sps)


    def resampy(self):
        import resampy
        temp_array = resampy.resample(self.signal[:], self.signal.sps, 2, axis=1, filter='kaiser_fast')
        self.signal.samples = temp_array
        self.signal.sps = 2
        return self.signal

if __name__ == '__main__':
    samples = np.load('./55new.npz')['arr_0']
    # samples_new = np.zeros_like(samples)
    # samples_new[0] = samples[0]
    # samples_new[1] = samples[1]
    # samples = samples_new
    receiver = CoherentReciver(samples,SignalParam(20e9,100e9/20e9,1550e-9),430,r'../experiment_library/txSymbols_qpsk.mat',100e9)
    receiver.dsp_processing()