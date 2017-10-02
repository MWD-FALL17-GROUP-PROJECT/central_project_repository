import numpy as np
from util import constants

"""
Multiplies the array by an identity matrix of the same .
Expects inputArray as a list.
Intention is to convert the sigma function to a matrix.
"""
def convertArrayToMatrix(inputArray):
    return np.matrix(inputArray) * np.matrix(np.eye(len(inputArray)))

def normalizer(min, max, value):
	return ((value - min + 0.0001) / (max - min + 0.0001))*constants.NORMALIZATION_VALUE
