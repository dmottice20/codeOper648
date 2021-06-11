import matplotlib.pyplot as plt
import numpy as np


def calculate_limiting_distribution(Pd):
    """
    :param Pd - induced DTMC from optimal policy
    :return pi_d - limiting distributino of Pd
    """
    # Right now -- this is for irreducible which is wrong.
    A_top_row = np.identity(Pd.shape[0])-Pd.transpose()
    A_bottom_row = np.repeat(1, Pd.shape[0])
    A = np.vstack((A_top_row, A_bottom_row))
    A = np.delete(A, 0, axis=0)
    b = np.vstack((np.asarray([np.repeat(0, Pd.shape[0]-1)]).transpose(), [1]))
    pi = np.matmul(np.linalg.inv(A), b)
    
    #pi = np.linalg.matrix_power(Pd, 1000)
    
    
    return pi
    