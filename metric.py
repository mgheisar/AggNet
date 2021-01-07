import numpy as np


def distEuclidean(vecA, vecB):
    # calculate Euclidean distance between two vectors or two data matrix
    return np.sqrt(np.sum((vecA - vecB) ** 2))


def distCosine(vecA, vecB):
    # calculate Cosine distance between two vectors
    return np.inner(vecA, vecB) / np.sqrt(np.inner(vecA, vecA) * np.inner(vecB, vecB))


def distPearson(vecA, vecB):
    # Pearson correlation
    N = len(vecA)
    meanA = np.mean(vecA)
    meanB = np.mean(vecB)
    devA = np.sqrt(np.sum((vecA - meanA) ** 2) / (N - 1))  # standard deviation of vecA
    devB = np.sqrt(np.sum((vecB - meanB) ** 2) / (N - 1))
    return (np.inner(vecA, vecB) - N * meanA * meanB) / ((N - 1) * devA * devB)
