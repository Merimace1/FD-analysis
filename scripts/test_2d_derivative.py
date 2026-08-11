from FD_analysis.basic_methods import *
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    low = 0
    high = 1
    N = 100
   
    domain = np.linspace(low,high,N)    

    xx, yy = np.meshgrid(domain,domain)


    func = np.sin(2*np.pi*xx) 

    x,y,func_derivative = twoD_derivative(domain,domain,func, axis = 1, method = "linear_interp",n=1)

    fig, axs = plt.subplots(1,2,figsize = (12,8))


    im = axs[0].pcolormesh(domain,domain, np.abs(func), cmap='viridis', shading='auto')
    im = axs[1].pcolormesh(x,y, np.abs(func_derivative), cmap='viridis', shading='auto')
    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label(label = "f(x,y)",rotation=270, labelpad=20)

    plt.show()