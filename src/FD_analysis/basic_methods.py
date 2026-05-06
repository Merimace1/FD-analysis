import numpy as np
import matplotlib.pyplot as plt

def twoD_derivative(x,y,matrix, axis =0) -> tuple:
    """
    Calculate the 2D derivative of a matrix along a specified axis.

    Parameters:
    matrix (numpy.ndarray): The input 2D array for which to calculate the derivative.
    axis (int): The axis along which to calculate the derivative. 
                Use 0 for vertical (y-axis) and 1 for horizontal (x-axis). Default is 0.

    Returns:
    numpy.ndarray: A 2D array containing the derivative values.
    """
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
