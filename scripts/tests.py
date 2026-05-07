from FD_analysis.basic_methods import *
import matplotlib.pyplot as plt
import numpy as np
import scipy 


if __name__ == "__main__":

    low = 0
    high = 20
    N = 100
    sampling = ( high - low )/N

    domain = np.linspace(low,high,N)    

    frequencies = np.fft.fftfreq(N,sampling)

    print("Nyquist frequency = ",1/2/sampling)
    print("maximum frequency displayed = ", np.max(np.abs(frequencies)))

    xx, yy = np.meshgrid(domain,domain)


    freq1 = 1
    freq2 = 1/5

    func = np.sin(freq1*2*np.pi*xx + freq1*2*np.pi*yy) + np.sin(freq2*2*np.pi*xx + freq2*2*np.pi*yy)


    #filter out bottom 10 percentile frequencies
    func_filt = Frequency_filtering(sampling,sampling,func,10, discard = "low", fill_value = 0)

    fig, axs = plt.subplots(1,2,figsize = (12,8))


    im = axs[0].pcolormesh(domain,domain, func_filt, cmap='viridis', shading='auto')
    im = axs[1].pcolormesh(domain,domain, func, cmap='viridis', shading='auto')
    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label('Rxx', rotation=270, labelpad=20)

    plt.show()