import numpy as np
class Equalizer(object):

    def __init__(self,signal,lr,ntaps,method,iter_number = 3,mode='training',training_time = 1):
        self.method = method
        if self.method.lower() not in ['cma','lms']:
            raise NotImplementedError

        self.lr = lr
        self.iter_number = iter_number

        self.ntaps = ntaps
        if divmod(self.ntaps,2)[1] ==0:
            self.ntaps = self.ntaps + 1

        if self.method =='cma':
            self.cma(signal)

        if self.method =='lms':
            self.mode = mode
            self.training_time = training_time
            self.lms(signal)

        if self.method =='lms_pll':
            self.lms_pll(signal)

        if self.method =='lms_superscaler':
            self.lms_super(signal)


        self.wxx = None
        self.wyy = None
        self.wxy = None
        self.wyx = None

        self.error_xpol = None
        self.error_ypol = None
        self.equlized_symbols = None

    def cma(self,signal):
        from myequalize import equalizer
        self.equlized_symbols,self.wxx,self.wxy,self.wyx,self.wyy,self.error_xpol,self.error_ypol = \
        equalizer(signal,os=signal.sps,ntaps=self.ntaps,mu=self.lr,iter_number=self.iter_number,method='cma')
    def lms(self,signal):
        from myequalize import equalizer
        self.equlized_symbols, self.wxx, self.wxy, self.wyx, self.wyy, self.error_xpol, self.error_ypol = \
            equalizer(signal, os=signal.sps, ntaps=self.ntaps, mu=self.lr, iter_number=self.iter_number,method='lms',mode=self.mode,training_time=self.training_time)
