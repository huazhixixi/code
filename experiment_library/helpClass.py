import numpy as np
import dataclasses
from scipy.constants import c


@dataclasses.dataclass
class Signal(object):
    '''
        samples: Electrial field of the signal
        sps: the samples per symbol
        center_wave_length: unit m
        symbol rate: unit Hz
    '''
    samples: np.ndarray
    sps: int
    center_wavelength: float
    symbol_rate: float

    @property
    def fs(self):
        return self.sps * self.symbol_rate

    @property
    def shape(self):
        return self.samples.shape

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value


@dataclasses.dataclass
class FiberParam(object):
    alpha: float = 0.2
    D: float = 16.7
    gamma: float = 1.3
    length: float = 80
    reference_wavelength: float = 1550
    slope: float = 0

    @property
    def alphalin(self):
        alphalin = self.alpha / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength * 1e-12 * self.D + (
                self.reference_wavelength * 1e-12) ** 2 * self.slope * 1e12)

        return res

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''

        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wavelength * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    def leff(self, length):
        '''

        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length
