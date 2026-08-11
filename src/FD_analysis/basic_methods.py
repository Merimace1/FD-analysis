import numpy as np
import matplotlib.pyplot as plt
import scipy

def twoD_derivative(x,y,matrix, axis =0,method = "Finite_diff",n=1) -> tuple:
    """
    Calculate the 2D derivative of a matrix along a specified axis.

    Parameters:
    matrix (numpy.ndarray): The input 2D array for which to calculate the derivative.
    axis (int): The axis along which to calculate the derivative. 
                Use 0 for vertical (y-axis) and 1 for horizontal (x-axis). Default is 0.

    Returns:
    numpy.ndarray: A 2D array containing the derivative values.
    """ 

    matrix = np.asarray(matrix)

    
    if method == "Finite_diff":
        if axis == 0:
            # Calculate vertical derivative
            y = y[1:]/2 + y[:-1]/2 
            return (x,y,np.diff(matrix, axis=0))
        elif axis == 1:
            # Calculate horizontal derivative
            x = x[1:]/2 + x[:-1]/2
            return (x,y,np.diff(matrix, axis=1))
        else:
            raise ValueError("Axis must be either 0 (vertical) or 1 (horizontal).")

    elif method == "linear_interp":
        if axis == 0:
            N,M = np.shape(matrix)
            result = np.zeros((N-2*n,M))

            for row_index in range(n,N-n):
                for column_index in range(M):

                    slope, offset = np.polyfit(y[row_index - n: row_index + n + 1],matrix[row_index - n: row_index + n + 1,column_index], 1)

                    result[row_index - n,column_index] = slope

            y = y[n:N-n]

            return (x,y,result)
                
        elif axis == 1:
            N,M = np.shape(matrix)
            result = np.zeros((N,M-2*n))

            for row_index in range(N):
                for column_index in range(n,M-n):

                    slope, offset = np.polyfit(x[column_index - n:column_index + n + 1],matrix[row_index,column_index - n:column_index + n + 1], 1)

                    result[row_index,column_index-n] = slope

            x = x[n:M-n]

            return (x,y,result)



def Frequency_filtering(vertical_sampling,horizontal_sampling,A,percetile, discard = "low", fill_value = 0) -> np.array:

    Nx, Ny = np.shape(A)

    vertical_freq = np.fft.fftfreq(Nx,vertical_sampling)
    horizontal_freq = np.fft.fftfreq(Ny,horizontal_sampling)

    threshold_vert = np.percentile(vertical_freq[:Nx//2], percetile)
    threshold_hor = np.percentile(horizontal_freq[:Ny//2], percetile)
    
    A_fft = scipy.fft.fft2(A)

    if discard == "low":

        ver_filt = np.abs(vertical_freq) >= threshold_vert
        hor_filt = np.abs(horizontal_freq) >= threshold_hor
    
    if discard == "high":
        ver_filt = np.abs(vertical_freq) <= threshold_vert
        hor_filt = np.abs(horizontal_freq) <= threshold_hor

    frequency_filter = np.logical_and.outer(ver_filt,hor_filt)

    A_filtered_fft = np.where(frequency_filter, A_fft, fill_value)
    A_filtered = scipy.fft.ifft2(A_filtered_fft)

    return np.real(A_filtered)


        



    
