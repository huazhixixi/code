import warnings
from typing import List
import numba
import tqdm
from numpy.fft import fftfreq
from scipy.constants import h, c
from scipy.fftpack import ifft, fft, next_fast_len
from scipy.special import erf

import numpy as np


class Edfa:

    def __init__(self, gain_db, nf, is_ase=True, mode='ConstantGain', expected_power=0):
        '''

        :param gain_db:
        :param nf:
        :param is_ase: 是否添加ase噪声
        :param mode: ConstantGain or ConstantPower
        :param expected_power: 当mode为ConstantPoower  时候，此参数有效
        '''

        self.gain_db = gain_db
        self.nf = nf
        self.is_ase = is_ase
        self.mode = mode
        self.expected_power = expected_power

    def one_ase(self, signal, gain_lin=None):
        '''

        :param signal:
        :return:
        '''
        lamb = signal.center_wavelength
        if gain_lin is None:
            one_ase = (h * c / lamb) * (self.gain_lin * self.nf_lin - 1) / 2
        else:
            one_ase = (h * c / lamb) * (gain_lin * self.nf_lin - 1) / 2
        return one_ase

    @property
    def gain_lin(self):
        return 10 ** (self.gain_db / 10)

    @property
    def nf_lin(self):
        return 10 ** (self.nf / 10)

    def traverse(self, signal):
        if self.mode == 'ConstantPower':
            #             raise NotImplementedError("Not implemented")
            signal_power = np.mean(np.abs(signal[0, :]) ** 2) + np.mean(
                np.abs(signal[1, :]) ** 2)
#             print(signal_power)
            desired_power_linear = (10 ** (self.expected_power / 10)) / 1000
            linear_gain = desired_power_linear / signal_power
            self.gain_db = 10*np.log10(linear_gain)
            signal[:] = np.sqrt(linear_gain) * signal[:]

        if self.mode == 'ConstantGain':
            signal[:] = np.sqrt(self.gain_lin) * signal[:]


        noise = self.one_ase(signal) * signal.fs_in_fiber
        each_pol_power = noise
        if self.is_ase:
            noise_sample = np.random.randn(*(signal[:].shape)) + 1j * np.random.randn(*(signal[:].shape))

            noise_sample = np.sqrt(each_pol_power / 2) * noise_sample
            signal[:] = signal[:]+noise_sample
        return signal

    def __call__(self, signal):
        self.traverse(signal)
        return signal

    def __str__(self):

        string = f"Model is {self.mode}\n" \
            f"Gain is {self.gain_db} db\n" \
            f"ase is {self.is_ase}\n" \
            f"noise figure is {self.nf}"
        return string

    def __repr__(self):
        return self.__str__()

class WSS(object):
    unit_dict = {'ghz':1,'hz':1e9}

    def __init__(self, frequency_offset, bandwidth, oft,unit):

        '''

        :param frequency_offset: value away from center [GHz]
        :param bandwidth: 3-db Bandwidth [Ghz]
        :param oft:GHZ
        '''
        self.frequency_offset = frequency_offset/WSS.unit_dict[unit.lower()]
        self.bandwidth = bandwidth/WSS.unit_dict[unit.lower()]
        self.oft = oft/WSS.unit_dict[unit.lower()]
        self.H = None
        self.freq = None

    def traverse(self, signal):

        sample = np.zeros_like(signal[:])
        for i in range(sample.shape[0]):
            sample[i, :] = signal[i, :]

        freq = fftfreq(len(sample[0, :]), 1 / signal.fs_in_fiber)
        freq = freq / 1e9
        self.freq = freq
        self.__get_transfer_function(freq)

        for i in range(sample.shape[0]):
            sample[i, :] = ifft(fft(sample[i, :]) * self.H)

        return sample

    def __call__(self, signal):
        sample = self.traverse(signal)
        signal[:] = sample
        return signal

    def __get_transfer_function(self, freq_vector):
        delta = self.oft / 2 / np.sqrt(2 * np.log(2))

        H = 0.5 * delta * np.sqrt(2 * np.pi) * (
                erf((self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta) - erf(
            (-self.bandwidth / 2 - (freq_vector - self.frequency_offset)) / np.sqrt(2) / delta))

        H = H / np.max(H)

        self.H = H

    def plot_transfer_function(self, freq=None):
        import matplotlib.pyplot as plt
        if self.H is None:
            self.__get_transfer_function(freq)
            self.freq = freq
        index = self.H > 0.001
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.scatter(self.freq[index], np.abs(self.H[index]), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("without log")
        plt.subplot(122)
        plt.scatter(self.freq[index], 10 * np.log10(np.abs(self.H[index])), color='b', marker='o')
        plt.xlabel('GHz')
        plt.ylabel('Amplitude')
        plt.title("with log")
        plt.show()

    def __str__(self):

        string = f'the center_frequency is {0 + self.frequency_offset}[GHZ] \t\n' \
            f'the 3-db bandwidth is {self.bandwidth}[GHz]\t\n' \
            f'the otf is {self.oft} [GHz] \t\n'
        return string

    def __repr__(self):
        return self.__str__()
