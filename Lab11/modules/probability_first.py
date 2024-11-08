import scipy

from modules.statistics import *
from math import log, pi
import numpy
from typing import Tuple

def logpdf_GAU_ND_singleSample(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()


def logpdf_GAU_ND_slow(X, mu, C):
    ret = []
    for i in range(X.shape[1]):
        ll = logpdf_GAU_ND_singleSample(X[:,i], mu, C)
        ret.append(ll)

    return np.array(ret).ravel()


def logpdf_GAU_ND_ale(X, mu, C):
    M = mu.shape[0]
    sign_log_det, log_det = numpy.linalg.slogdet(C)
    diff = X - mu.reshape((mu.size, 1))
    inner_term = numpy.dot(numpy.dot(diff.T, numpy.linalg.inv(C)), diff)
    log_densities = -0.5 * (M * numpy.log(2 * numpy.pi) + log_det + inner_term.diagonal())

    return log_densities


def logpdf_GAU_ND(x: np.array, mu: np.array, C):
    """ Returns the log of the probability density function of a multivariate Gaussian distribution
        x: may be a single sample or a matrix of samples, and the number of features can be however much. """
    if mu.ndim != 1:
        if mu.ndim == 2 and mu.shape[1] != 1:
            raise ValueError("mu must be a column vector")
        elif mu.ndim > 2:
            raise ValueError("mu must be a 1D numpy array or a 2D numpy column vector. Its shape is " + str(mu.shape))

    # If x is a single sample, make it a 2D np column vector, else it will remain a generic 2D np array
    x = to_numpy_matrix(x, "col")
    mu = to_numpy_matrix(mu, "col")

    M = how_many_features(x)
    (_, SigmaLogdet) = np.linalg.slogdet(C)
    # print("C in logpdf_GAU_ND is:")
    # print(C)
    SigmaInv = np.linalg.inv(C)

    # Don't put any sign yet (i.e. don't multiply by -1 yet)
    first_term = (M / 2) * log(2*pi)
    second_term = (1 / 2) * SigmaLogdet

    # # If x was always just 1dimensional
    # third_term = (1 / 2) * ( (x - mu).T @ SigmaInv @ (x - mu) )

    # Since I only need the diagonal of (x - mu)T @ SigmaInv @ (x - mu) I can exploit element wise multiplication
    third_term_part1 = ( (x - mu).T @ SigmaInv ) * (x - mu).T
    # Now if I sum over row i I have element ii of (x - mu)T @ SigmaInv @ (x - mu), i.e. it's i-th diagonal element
    third_term_part2 = np.sum(third_term_part1, axis=1).T
    third_term = (1 / 2) * third_term_part2

    return (-1 * first_term) + (-1 * second_term) + (-1 * third_term)


def loglikelihood(x, mu, C):
    log_densities = logpdf_GAU_ND(x, mu, C)
    return np.sum(vrow(log_densities), axis=1)


""" llr is a vector containing the log likelihood ratio sample by sample """
def get_llrs(logscore):
    if logscore.shape[0] != 2:
        raise ValueError("loglikelihood matrix for binary classification has more than two classes")

    return logscore[1, :] - logscore[0, :]
