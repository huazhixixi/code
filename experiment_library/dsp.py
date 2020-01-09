import numpy as np
from scipy.fftpack import fftfreq
from scipy.constants import c
import copy
import matplotlib.pyplot as plt

from helpClass import Signal

plt.ion()


def remove_dc(signal):
    signal_samples = np.atleast_2d(signal[:])
    signal_samples = signal_samples - np.mean(signal_samples, axis=1, keepdims=True)

    if isinstance(signal, Signal):
        signal.samples = signal_samples
        return signal
    else:

        return signal_samples


def resampling(signal, adc_rate, symbol_rate, new_sps=2):
    from resampy import resample
    signal_samples = np.atleast_2d(signal[:])
    # assert signal_samples.shape[0] == 1
    ori_sps = adc_rate / symbol_rate
    signal_samples = resample(signal_samples[:], ori_sps, new_sps, axis=1)

    if isinstance(signal, Signal):
        signal.sps = new_sps
        signal.samples = signal_samples
        return signal
    else:
        return signal_samples


def orthonormalize_signal(signal, os=1):
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].
    Parameters
    ----------
    E : array_like
       input signal
    os : int, optional
        oversampling ratio of the signal
    Returns
    -------
    E_out : array_like
        orthonormalized signal
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """

    E = np.atleast_2d(signal[:])
    E_out = np.empty_like(E)
    for l in range(E.shape[0]):
        # Center
        real_out = E[l, :].real - E[l, :].real.mean()
        tmp_imag = E[l, :].imag - E[l, :].imag.mean()

        # Calculate scalar products
        mean_pow_inphase = np.mean(real_out ** 2)
        mean_pow_quadphase = np.mean(tmp_imag ** 2)
        mean_pow_imb = np.mean(real_out * tmp_imag)

        # Output, Imag orthogonal to Real part of signal
        sig_out = real_out / np.sqrt(mean_pow_inphase) + \
                  1j * (tmp_imag - mean_pow_imb * real_out / mean_pow_inphase) / np.sqrt(mean_pow_quadphase)
        # Final total normalization to ensure IQ-power equals 1
        E_out[l, :] = sig_out - np.mean(sig_out[::os])
        E_out[l, :] = E_out[l, :] / np.sqrt(np.mean(np.abs(E_out[l, ::os]) ** 2))
    if isinstance(signal,Signal):
        signal.samples = E_out
        return signal
    else:
        return E_out


def cd_compensation(signal, spans, inplace=False):
    '''

    This function is used for chromatic dispersion compensation in frequency domain.
    The signal is Signal object, and a new sample is created from property data_sample_in_fiber

    :param signal: Signal object
    :param spans: Span object, the signal's has propagated through these spans
    :param inplace: if True, the compensated sample will replace the original sample in signal,or new ndarray will be r
    eturned

    :return: if inplace is True, the signal object will be returned; if false the ndarray will be returned
    '''
    try:
        import cupy as np
    except Exception:
        import numpy as np

    center_wavelength = signal.center_wavelength
    freq_vector = fftfreq(signal[0, :].shape[0], 1 / signal.fs)
    omeg_vector = 2 * np.pi * freq_vector

    sample = np.array(signal[:])

    if not isinstance(spans, list):
        spans = [spans]

    for span in spans:
        beta2 = -span.beta2(center_wavelength)
        dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span.length
        dispersion = np.array(dispersion)
        for pol_index in range(sample.shape[0]):
            sample[pol_index, :] = np.fft.ifft(np.fft.fft(sample[pol_index, :]) * np.exp(dispersion))

    if inplace:
        if hasattr(np, 'asnumpy'):
            sample = np.asnumpy(sample)
        signal[:] = sample
        return signal
    else:
        if hasattr(np, 'asnumpy'):
            sample = np.asnumpy(sample)
        signal = copy.deepcopy(signal)
        signal[:] = sample
        return signal


def syncsignal(symbol_tx, rx_signal, sps, visable=False):
    '''

        :param symbol_tx: 发送符号
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果

        # 不会改变原信号

    '''
    from scipy.signal import correlate
    symbol_tx = np.atleast_2d(symbol_tx)
    sample_rx = np.atleast_2d(rx_signal[:])
    out = np.zeros_like(sample_rx)
    # assert sample_rx.ndim == 1
    # assert symbol_tx.ndim == 1
    assert sample_rx.shape[1] >= symbol_tx.shape[1]
    for i in range(symbol_tx.shape[0]):
        symbol_tx_temp = symbol_tx[i, :]
        sample_rx_temp = sample_rx[i, :]

        res = correlate(sample_rx_temp[::sps], symbol_tx_temp)
        if visable:
            plt.figure()
            plt.plot(np.abs(np.atleast_2d(res)[0]))
            plt.show()
        index = np.argmax(np.abs(res))

        out[i] = np.roll(sample_rx_temp, sps * (-index - 1 + symbol_tx_temp.shape[0]))
    if isinstance(rx_signal,Signal):
        rx_signal.samples = out
        return rx_signal
    else:
        return out


def syncsignal_tx2rx(symbol_rx, symbol_tx):
    from scipy.signal import correlate

    symbol_tx = np.atleast_2d(symbol_tx)
    symbol_rx = np.atleast_2d(symbol_rx)
    out = np.zeros_like(symbol_tx)
    # assert sample_rx.ndim == 1
    # assert symbol_tx.ndim == 1
    assert symbol_tx.shape[1] >= symbol_rx.shape[1]
    for i in range(symbol_tx.shape[0]):
        symbol_tx_temp = symbol_tx[i, :]
        sample_rx_temp = symbol_rx[i, :]

        res = correlate(symbol_tx_temp, sample_rx_temp)

        index = np.argmax(np.abs(res))

        out[i] = np.roll(symbol_tx_temp, -index - 1 + sample_rx_temp.shape[0])
    return out
