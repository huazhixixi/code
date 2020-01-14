# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:18:28 2019

@author: shang
"""

import numpy as np
import numba
from numba import prange
from scipy.signal import lfilter

from .helpClass import Signal


def find_freq_offset(sig, fs,average_over_modes = True, fft_size = 2**18):
    """
    Find the frequency offset by searching in the spectrum of the signal
    raised to 4. Doing so eliminates the modulation for QPSK but the method also
    works for higher order M-QAM.

    Parameters
    ----------
        sig : array_line
            signal array with N modes
        os: int
            oversampling ratio (Samples per symbols in sig)
        average_over_modes : bool
            Using the field in all modes for estimation

        fft_size: array
            Size of FFT used to estimate. Should be power of 2, otherwise the
            next higher power of 2 will be used.

    Returns
    -------
        freq_offset : int
            found frequency offset

    """
    if not((np.log2(fft_size)%2 == 0) | (np.log2(fft_size)%2 == 1)):
        fft_size = 2**(int(np.ceil(np.log2(fft_size))))

    # Fix number of stuff
    sig = np.atleast_2d(sig)
    npols, L = sig.shape

    # Find offset for all modes
    freq_sig = np.zeros([npols,fft_size])
    for l in range(npols):
        freq_sig[l,:] = np.abs(np.fft.fft(sig[l,:]**4,fft_size))**2

    # Extract corresponding FO
    freq_offset = np.zeros([npols,1])
    freq_vector = np.fft.fftfreq(fft_size,1/fs)/4
    for k in range(npols):
        max_freq_bin = np.argmax(np.abs(freq_sig[k,:]))
        freq_offset[k,0] = freq_vector[max_freq_bin]


    if average_over_modes:
        freq_offset = np.mean(freq_offset)

    return freq_offset

def comp_freq_offset(sig, freq_offset, os=1 ):
    """
    Compensate for frequency offset in signal

    Parameters
    ----------
        sig : array_line
            signal array with N modes
        freq_offset: array_like
            frequency offset to compensate for if 1D apply to all modes
        os: int
            oversampling ratio (Samples per symbols in sig)


    Returns
    -------
        comp_signal : array with N modes
            input signal with removed frequency offset

    """
    # Fix number of stuff
    ndim = sig.ndim
    sig = np.atleast_2d(sig)
    #freq_offset = np.atleast_2d(freq_offset)
    npols = sig.shape[0]

    # Output Vector
    comp_signal = np.zeros([npols, np.shape(sig)[1]], dtype=sig.dtype)

    # Fix output
    sig_len = len(sig[0,:])
    time_vec = np.arange(1,sig_len + 1,dtype = float)
    for l in range(npols):
        lin_phase = 2 * np.pi * time_vec * freq_offset[l] /  os
        comp_signal[l,:] = sig[l,:] * np.exp(-1j * lin_phase)
    if ndim == 1:
        return comp_signal.flatten()
    else:
        return comp_signal
    
    
def segment_find_freq(signal,fs,group,apply=True):
    from .dsp_tool import __segment_axis
    from .dsp_tool import get_time_vector
    
    length = signal.shape[1]//group
    freq_offset = []
    
    if length * group != signal.shape[1]:
        import warnings
        warnings.warn("The group can not be divided into integers and some points will be discarded")
        
    time_vector = get_time_vector(signal.shape[1],fs)
    time_vector = np.atleast_2d(time_vector)[0]
    
    last_point = 0
    
    xpol = __segment_axis(signal[0], length, 0)
    ypol = __segment_axis(signal[1], length, 0)
    time_vector_segment = time_vector[:length]
    phase = np.zeros_like(xpol)
    
    for idx,(xpol_row,ypol_row) in enumerate(zip(xpol,ypol)):
        array = np.array([xpol_row,ypol_row])      
        freq = find_freq_offset(array,fs,fft_size=xpol.shape[1])
        phase[idx] = 2*np.pi*freq*time_vector_segment+last_point
        freq_offset.append(freq)
        last_point = phase[idx,-1]
    if apply:
        xpol = xpol * np.exp(-1j*phase)
        ypol = ypol * np.exp(-1j*phase)
    
    xpol = xpol.flatten()
    ypol = ypol.flatten()

    if isinstance(signal,Signal):
        signal.samples = np.array([xpol,ypol])
        return signal,freq_offset
    else:
        return np.array([xpol,ypol]),freq_offset
        

def superscalar(symbol_in, training_symbol, block_length, pilot_number, constl, g,filter_n=20):
    # delete symbols to assure the symbol can be divided into adjecent channels
    symbol_in = np.atleast_2d(symbol_in)
    training_symbol =np.atleast_2d(training_symbol)
    constl = np.atleast_2d(constl)
    assert symbol_in.shape[0]==1
    assert training_symbol.shape[0]==1
    assert constl.shape[0]==1
    divided_symbols, divided_training_symbols = __initialzie_superscalar(symbol_in, training_symbol, block_length)
    angle = __initialzie_pll(divided_symbols, divided_training_symbols, pilot_number)

    divided_symbols = divided_symbols * np.exp(-1j * angle)
    divided_symbols = first_order_pll(divided_symbols, (constl), g)
    divided_symbols[0::2, :] = divided_symbols[0::2, ::-1]
    divided_symbols = divided_symbols.reshape((1, -1))
    # ml
    decision_symbols = np.zeros(divided_symbols.shape[1],dtype=np.complex)
    exp_decision(divided_symbols[0,:],constl[0,:],decision_symbols)

    hphase_ml = symbol_in[0,:len(decision_symbols)]/decision_symbols
    hphase_ml = np.atleast_2d(hphase_ml)
    h = np.ones((1,2*filter_n+1))
    hphase_ml = lfilter(h[0,:],1,hphase_ml)
    hphase_ml = np.roll(hphase_ml,-filter_n,axis=1)
    phase_ml = np.angle(hphase_ml)
    divided_symbols = symbol_in[:,:len(decision_symbols)] * np.exp(-1j*phase_ml)
    #ml completed
    divided_training_symbols[0::2, :] = divided_training_symbols[0::2, ::-1]
    divided_training_symbols = divided_training_symbols.reshape((1, -1))
    # scatterplot(divided_symbols,False,'pyqt')

    # filrst order pll

    return divided_symbols, divided_training_symbols,phase_ml


@numba.jit(nopython=True, parallel=True)
def first_order_pll(divided_symbols, constl, g):
    constl = np.atleast_2d(constl)
    phase = np.zeros((divided_symbols.shape[0], divided_symbols.shape[1]))
    for i in prange(divided_symbols.shape[0]):
        signal = divided_symbols[i, :]
        each_error = phase[i, :]
        for point_index, point in enumerate(signal):
            if point_index == 0:
                point = point * np.exp(-1j * 0)
            else:
                point = point * np.exp(-1j * each_error[point_index - 1])
            point_decision = decision(point, constl)
            signal[point_index] = point
            point_decision_conj = np.conj(point_decision)
            angle_difference = np.angle(point * point_decision_conj)

            if point_index > 0:
                each_error[point_index] = angle_difference * g + each_error[point_index - 1]
            else:
                each_error[point_index] = angle_difference * g

    return divided_symbols


def __initialzie_pll(divided_symbols, divided_training_symbols, pilot_number):
    '''
        There are pilot_number symbols of each row,the two adjecnt channel use the same phase,because they are simillar

    '''
    # get pilot symbol
    pilot_signal = divided_symbols[:, : pilot_number]
    pilot_traing_symbol = divided_training_symbols[:, :pilot_number]

    angle = (pilot_signal / pilot_traing_symbol)
    angle = angle.flatten()
    angle = angle.reshape(-1, 2 * pilot_number)
    angle = np.sum(angle, axis=1, keepdims=True)
    angle = np.angle(angle)

    angle_temp = np.zeros((angle.shape[0] * 2, angle.shape[1]), dtype=np.float)
    angle_temp[0::2, :] = angle
    angle_temp[1::2, :] = angle_temp[0::2, :]
    return angle_temp


def __initialzie_superscalar(symbol_in, training_symbol, block_length):
    # delete symbols to assure the symbol can be divided into adjecent channels
    symbol_in = np.atleast_2d(symbol_in)
    training_symbol = np.atleast_2d(training_symbol)
    assert symbol_in.shape[0] == 1
    symbol_length = len(symbol_in[0, :])
    assert divmod(block_length, 2)[1] == 0

    if divmod(symbol_length, 2)[1] != 0:
        # temp_symbol = np.zeros((symbol_in.shape[0], symbol_in.shape[1] - 1), dtype=np.complex)
        # temp_training_symbol = np.zeros((training_symbol.shape[0], training_symbol.shape[1] - 1), dtype=np.complex)
        temp_symbol = symbol_in[:, :-1]
        temp_training_symbol = training_symbol[:, :-1]
    else:
        temp_symbol = symbol_in
        temp_training_symbol = training_symbol

    # divide into channels
    channel_number = int(len(temp_symbol[0, :]) / block_length)
    if divmod(channel_number, 2)[1] == 1:
        channel_number = channel_number - 1
    divided_symbols = np.zeros((channel_number, block_length), dtype=np.complex)
    divided_training_symbols = np.zeros((channel_number, block_length), dtype=np.complex)
    for cnt in range(channel_number):
        divided_symbols[cnt, :] = temp_symbol[0, cnt * block_length:(cnt + 1) * block_length]
        divided_training_symbols[cnt, :] = temp_training_symbol[0, cnt * block_length:(cnt + 1) * block_length]
        if divmod(cnt, 2)[1] == 0:
            divided_symbols[cnt, :] = divided_symbols[cnt, ::-1]
            divided_training_symbols[cnt, :] = divided_training_symbols[cnt, ::-1]
    #             print(divided_training_symbols.shape)
    # First Order PLL

    return divided_symbols, divided_training_symbols

@numba.guvectorize([(numba.types.complex128[:],numba.types.complex128[:],numba.types.complex128[:])], '(n),(m)->(n)',nopython=True)
def exp_decision(symbol,const,res):
    for index,sym in enumerate(symbol):
        # distance = sym-const
        distance = np.abs(sym-const)
        res[index] = const[np.argmin(distance)]




@numba.jit(cache=True)
def decision(symbol, constl):
    '''
        constl must be 2d
    '''
    constl = np.atleast_2d(constl)
    distance = np.abs(constl[0] - symbol)
    decision = constl[0, np.argmin(distance)]

    return decision



