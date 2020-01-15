import pandas as pd
import numpy as np
import os

def read_data(BASE_DIR,saveto):
    for file_name in os.listdir(BASE_DIR):
        if 'Ch1' in file_name:
            ch1_dir = BASE_DIR+'/'+file_name
        if 'Ch2' in file_name:
            ch2_dir = BASE_DIR+'/' + file_name
        if 'Ch3' in file_name:
            ch3_dir = BASE_DIR+'/' + file_name
        if 'Ch4' in file_name:
            ch4_dir = BASE_DIR+'/' + file_name


    ch1 = pd.read_csv(ch1_dir, header=None,usecols=[4])
    ch1 = ch1.values

    ch2 = pd.read_csv(ch2_dir,  header=None,usecols=[4])
    ch2 = ch2.values

    ch3 = pd.read_csv(ch3_dir, header=None,usecols=[4])
    ch3 = ch3.values

    ch4 = pd.read_csv(ch4_dir, header=None,usecols=[4])
    ch4 = ch4.values

    ch1.shape = 1, -1
    ch2.shape = 1, -1
    ch3.shape = 1, -1
    ch4.shape = 1, -1
    ch2 = np.roll(ch2, -1, axis=1)
    ch4 = np.roll(ch4, 2, axis=1)

    xpol = ch1 + 1j * ch2
    ypol = ch3 + 1j * ch4
    from scipy.io import savemat
    # savemat(saveto,dict(samples = np.vstack((xpol,ypol))))
    np.savez(saveto,np.vstack((xpol,ypol)))


if __name__ == '__main__':
    for ith,dir in enumerate(os.listdir('/Volumes/D/ai/5dbm')):
        if not dir.startswith('.'):
            read_data('/Volumes/D/ai/5dbm/'+dir,f'./npzdata/5dbm_{ith}.npz')