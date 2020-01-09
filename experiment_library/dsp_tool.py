# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:16:10 2019

@author: shang
"""

import numpy as np

import numba

from helpClass import Signal


def __segment_axis(a, length, overlap, mode='cut', append_to_end=0):
    """
        Generate a new array that chops the given array along the given axis into
        overlapping frames.

        example:
        >>> segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])

        arguments:
        a       The array to segment must be 1d-array
        length  The length of each frame
        overlap The number of array elements by which the frames should overlap

        end     What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:

                'cut'   Simply discard the extra values
                'pad'   Pad with a constant value

        append_to_end:    The value to use for end='pad'

        a new array will be returned.

    """

    if a.ndim !=1:
        raise Exception("Error, input array must be 1d")
    if overlap > length:
        raise Exception("overlap cannot exceed the whole length")

    stride = length - overlap
    row = 1
    total_number = length
    while True:
        total_number = total_number + stride
        if total_number > len(a):
            break
        else:
            row = row + 1

    # 一共要分成row行
    if total_number > len(a):
        if mode == 'cut':
            b = np.zeros((row, length), dtype=np.complex128)
            is_append_to_end = False
        else:
            b = np.zeros((row + 1, length), dtype=np.complex128)
            is_append_to_end = True
    else:
        b = np.zeros((row, length), dtype=np.complex128)
        is_append_to_end = False

    index = 0
    for i in range(row):
        b[i, :] = a[index:index + length]
        index = index + stride

    if is_append_to_end:
        last = a[index:]

        b[row, 0:len(last)] = last
        b[row, len(last):] = append_to_end

    return b




def get_time_vector(length,fs):
    
    dt = 1/fs
    time_vector = dt *np.arange(length)
    return np.atleast_2d(time_vector)





def normalise_and_center(signal):
    def cabssquared(x):
        """Calculate the absolute squared of a complex number"""
        return x.real ** 2 + x.imag ** 2
    E = signal[:]
    if E.ndim > 1:
        E = E - np.mean(E, axis=-1)[:, np.newaxis]
        P = np.sqrt(np.mean(cabssquared(E), axis=-1))
        E /= P[:, np.newaxis]
    else:
        E = E.real - np.mean(E.real) + 1.j * (E.imag - np.mean(E.imag))
        P = np.sqrt(np.mean(cabssquared(E)))
        E /= P
    if isinstance(signal,Signal):
        signal.samples = E
        return signal
    else:
        return E


@numba.njit(cache=True)
def decision(symbol, constl):
    '''
        constl must be 2d
    '''
    constl = np.atleast_2d(constl)
    distance = np.abs(constl[0] - symbol)
    decision = constl[0, np.argmin(distance)]
    return decision