from scipy.constants import c
import numpy as np


class Laser(object):

    def __init__(self, linewidth, center_frequency, laser_power, is_phase_noise=False):
        '''

        :param linewidth: The linewidth of the laser [Hz]
        :param center_frequency: The center frequency of the laser [Hz]
        :param is_phase_noise: if there is phase noise or not True/False
        :param laser_power: the power of the fiber [dBm]
        '''
        self.linewidth = linewidth
        self.center_frequency = center_frequency
        self.is_phase_noise = is_phase_noise
        self.laser_power = laser_power

    def prop(self, signal):
        signal[:] = signal[:] / np.sqrt(np.mean(np.abs(signal[:]) ** 2, axis=1, keepdims=True))
        laser_power = 10 ** (self.laser_power / 10)
        laser_power = laser_power / 1000
        signal[:] = np.sqrt(laser_power) * signal[:]

        if self.is_phase_noise:
            sigma2 = 2 * np.pi * self.linewidth / signal.fs_in_fiber

            initial_phase = np.pi * (2 * np.random.rand(1) - 1)
            dtheta = np.zeros(signal.shape)
            dtheta = np.atleast_2d(dtheta)
            for row in dtheta:
                row[1:] = np.sqrt(sigma2) * np.random.randn(1, len(row) - 1)

            phase_noise = initial_phase + np.cumsum(dtheta, axis=1)
            signal[:] = signal[:] * np.exp(1j * phase_noise)
            return signal

from collections import namedtuple

ConstantGainEdfaParam = namedtuple('param','expected_gain nf')
ConstantPowerEdfaParam = namedtuple('param','expected_power nf')

class Edfa(object):

    def __init__(self,mode,is_ase_noise):
        '''

        :param mode:
        :param param_dict:
        '''
        self.mode = mode
        self.is_ase_noise = is_ase_noise
        self.gain = None
        self.nf = None


    def get_noise_seq(self,pol_number,length,wavelength,fs_in_fiber):
        '''

        :param length: The length of the signal
        :param wavelength: The wavelength: [m]
        :return:
        '''
        from scipy.constants import c,h
        ase_psd_one_polarization = (self.gain - 1)*10**(self.nf/10)/2*(h*c/wavelength)
        ase_power = ase_psd_one_polarization * fs_in_fiber # one pol ase power
        # divded by 2 because of the complex noise
        noise_sequence = np.sqrt(ase_power/2) * (np.random.randn(pol_number,length) + 1j * np.random.randn(pol_number,length))
        return noise_sequence


    def prop(self,signal):
        pol_number = np.atleast_2d(signal[:]).shape[0]
        length = np.atleast_2d(signal[:]).shape[1]
        self.gain = 10**(self.gain/10)
        signal[:] = np.sqrt(self.gain) * signal[:]
        if self.is_ase_noise:
            noise_sequence = self.get_noise_seq(signal,pol_number,length,signal.wavelength,signal.fs_in_fiber)
            signal[:] = signal[:] + noise_sequence
        return signal

class ConstantGainEdfa(Edfa):

    def __init__(self,param:ConstantGainEdfaParam,is_ase_noise):
        '''

        :param param: expected_gain:dB,nf:dB
        :param is_ase_noise: False or True
        '''
        super().__init__(mode='ConstantGain',is_ase_noise = is_ase_noise)
        self.gain = param.expected_gain
        self.nf = param.nf



class ConstantPowerEdfa(Edfa):

    def __init__(self,param:ConstantPowerEdfaParam,is_ase_noise,signal):
        '''

        :param param: expected_power: dBm,nf:dB
        :param is_ase_noise: False or True
        '''
        super().__init__(mode='ConstantPower',is_ase_noise = is_ase_noise)
        self.expected_power = param.expected_power
        self.nf = param.nf
        self.calc_gain(signal)

    def calc_gain(self,signal):
        signal_samples = np.atleast_2d(signal[:])
        signal_power = np.sum(np.sum(np.abs(signal_samples) ** 2, axis=1))
        signal_power = 10*np.log10(signal_power * 1000)
        self.gain = signal_power - self.expected_power


class WSS(object):

    def __init__(self,bw,otf):
        '''

        :param bw: [hz]
        :param otf: [hz]
        '''
        self.bw = bw
        self.otf = otf

    def prop(self,signal):

        pass

class IQ(object):
    pass

