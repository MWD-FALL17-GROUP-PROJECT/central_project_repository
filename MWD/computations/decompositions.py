from sklearn.utils.extmath import randomized_svd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import pandas as pd
import util.constants


def PCADecomposition(inputMatrix, n_components):
	mat_std = StandardScaler().fit_transform(inputMatrix,inputMatrix.columns.values)
	pca = decomposition.PCA(n_components)
	pca.fit(mat_std)
	return pca.components_


def PCADimensionReduction(inputMatrix,new_dimensions):
	mat_std = StandardScaler().fit_transform(inputMatrix, inputMatrix.columns.values)
	pca = decomposition.PCA(new_dimensions)
	pca.fit(mat_std)
	return pca.transform(mat_std)

"""
inputMatrix is assumed to be in the pandas format
n_componenets is the top latent semantics we need. The Sigma matrix will have these many values in its list
Sigma is a list of values and not a matrix
"""
def SVDDecomposition(inputMatrix, n_components):
    mat_std = StandardScaler().fit_transform(inputMatrix,inputMatrix.columns.values)
    U, Sigma, VT = randomized_svd(mat_std,
                              n_components,
                              n_iter=util.constants.ITERATIONS,
                              random_state=util.constants.RANDOM_STATE)
    return U, Sigma, VT

