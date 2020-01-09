import numpy as np
import matplotlib.pyplot as plt
from simulation_code.SignalDefine import QamSignal


signal = QamSignal(sps=2,sps_in_fiber=4,rrc_param={'roll_off':0.01},length=65536,
                   order=16,power=0,baudrate=35e9,center_frequency=193.1e12)

signal.prepare()
