import numpy as np
import matplotlib.pyplot as plt
from simulation_code.SignalDefine import QamSignal
from simulation_code.dsp import MatchedFilter
from simulation_code.optics.optics import Laser
from experiment_library.myequalize import equalizer
signal = QamSignal(sps=2,sps_in_fiber=4,rrc_param={'roll_off':0.01},length=65536,
                   order=16,power=0,baudrate=35e9,center_frequency=193.1e12)

signal = signal.prepare()
laser = Laser(100e3,193.1e12,0,True)
laser.prop(signal)

# resampling
from resampy import resample
signal.ds_in_fiber = resample(signal.ds_in_fiber,signal.sps_in_fiber,2,axis=1,filter='kaiser_fast')
signal.sps_in_fiber = 2
#matched filter
matched_filter = MatchedFilter(0.01,signal.sps_in_fiber,1024)
signal = matched_filter(signal)

signal[:] = signal[:]/np.sqrt(np.mean(np.abs(signal[:])**2,axis=1,keepdims=True))
from experiment_library.phase import superscalar
from experiment_library.myequalize import equalizer
from scipy.io import loadmat
signal[:] = loadmat('rx.mat')['rx_samples']
signal.symbol = loadmat('rx.mat')['tx_symbols']
symbol, (wxx, wxy, wyx, wyy, error_xpol, error_ypol) = equalizer(signal[:],2,31,0.001,3,'lms',training_time=3,train_symbol=signal.symbol)
import matplotlib.pyplot as plt
plt.scatter(symbol[0].real,symbol[0].imag)
# sysmbols,train_symbol,_ = superscalar(signal[0,::2],signal.symbol[0],200,8,np.unique(signal.symbol[0]),0.02)
# import matplotlib.pyplot as plt
# plt.scatter(sysmbols[0].real,sysmbols[0].imag)
