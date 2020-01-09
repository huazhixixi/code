def power_meter(samples):
    pass


def electrical_spectrum(samples):
    import matplotlib.pyplot as plt
    import numpy as np
    samples = np.atleast_2d(samples)
    fig, ax = plt.subplots(1, samples.shape[0])
    ax = np.atleast_2d(ax)[0]

    for row_index in range(samples.shape[0]):
        ax[row_index].psd(samples[row_index], NFFT=16384, fs=samples.fs_in_fiber)

    plt.show()
    plt.tight_layout()


def osa(samples, resolution):
    pass
