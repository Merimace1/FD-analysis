from FD_analysis.basic_methods import *
import numpy as np

a = np.arange(9).reshape(3,3)
a[1,:] = a[1,:]+ 1
print(np.diff(a, axis=0))