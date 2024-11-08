import pickle

import numpy as np
import scipy
from tqdm import tqdm

from modules.common_matrix_operations import to_numpy_matrix, vcol
from modules.probability_first import logpdf_GAU_ND
from modules.statistics import how_many_samples, get_covariance_matrix


class GMMParameters:
    components: list

    def __init__(self, components: list):
        self.components = components

    def get_components(self) -> list:
        return self.components

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance


def logpdf_GMM(X: np.array, gmm: np.array) -> np.array:
    """ Returns the marginals and the joint log-likelihoods of the samples in X, given the GMM model. """
    if X.ndim > 2:
        raise ValueError("X must be at most a 2D numpy array")
    if not isinstance(gmm, list):
        raise ValueError("gmm must be a python list. It contains as many tuples (filled with the parameters "
                         "of the component) as there are GMM components")

    X = to_numpy_matrix(X, "col")

    logscore_joint = np.zeros((len(gmm), how_many_samples(X)))
    for g, (w, mu, C) in enumerate(gmm):
        logscore_joint[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    #                                                            somma verticalmente
    logscore_marginals = scipy.special.logsumexp(logscore_joint, axis=0)  # As many as there are samples

    # What we call logscore_marginals are actually the loglikelihoods of the GMM
    # When we call them marginals, it is because the same term is used in the denominator of the posterior when looking
    # for the posterior of cluster C given a sample
    return logscore_marginals.ravel(), logscore_joint

def EMM(D, GMM_params, delta=1e-6, variant="std", min_cov=0.01):
    if variant not in ["std", "diag", "tied"]:
        raise ValueError("The variant must be either 'std', 'diag' or 'tied'")

    # First iteration computed now, as the update is better done at the end of the previous loop, instead
    # of at the beginning of the next one
    """ E-step t """
    # 1) First of all apply the minimum covariance constraint to the starting params too
    for g, (w_g, mu_g, C_g) in enumerate(GMM_params):
        U, s, _ = np.linalg.svd(C_g)
        s[s < min_cov] = min_cov
        C_g = np.dot(U, vcol(s) * U.T)
        GMM_params[g] = (w_g, mu_g, C_g)

    # 2) Now get the gammas
    logscore_marginals, logscore_joint = logpdf_GMM(D, GMM_params)
    logscore_posterior = logscore_joint - logscore_marginals  # Broadcasting
    gammas = np.exp(logscore_posterior)  # GxN np.array where G is the number of clusters and N is the number of samples
    last_loglikelihood = np.average(logscore_marginals)
    # print("ll at step 0: " + str(last_loglikelihood))

    t = 1
    good_enough = False
    while not good_enough:
        """ M-step t """
        Z = np.sum(gammas, axis=1)
        F = np.dot(gammas, D.T)
        S = np.zeros((len(GMM_params), D.shape[0], D.shape[0]))

        # Could optimize this using broadcasting like I did for logpdf_GAU_ND, but it's a hassle.
        for g in range(len(GMM_params)):
            for i in range(D.shape[1]):
                S[g] += gammas[g, i] * vcol(D[:, i:i+1]) @ vcol(D[:, i:i+1]).T

        for g, _ in enumerate(GMM_params):
            Z_g = Z[g]
            F_g = F[g]
            S_g = S[g]

            # Calculate the mu_g of iteration t+1
            mu_g = F_g / Z_g

            # Calculate the C_g of iteration t+1
            C_g = S_g / Z_g - (vcol(mu_g) @ vcol(mu_g).T)  # uses mu_g that has just been updated
            if variant == "diag":
                C_g = C_g * np.eye(C_g.shape[0])

            # Calculate the w_g of iteration t+1
            w_g = Z_g / D.shape[1]

            GMM_params[g] = (w_g, mu_g, C_g)

        if variant == "tied":
            tiedC = 0
            # Common Cov computation
            for g, (w_g, mu_g, C_g) in enumerate(GMM_params):
                tiedC += w_g * C_g

            # -- Minimum Cov constraint
            U, s, _ = np.linalg.svd(tiedC)
            s[s < min_cov] = min_cov
            tiedC = np.dot(U, vcol(s) * U.T)

            for g, (w_g, mu_g, C_g) in enumerate(GMM_params):
                GMM_params[g] = (w_g, mu_g, tiedC)
        else:
            # -- Minimum Cov constraint
            for g, (w_g, mu_g, C_g) in enumerate(GMM_params):
                U, s, _ = np.linalg.svd(C_g)
                s[s < min_cov] = min_cov
                C_g = np.dot(U, vcol(s) * U.T)

                GMM_params[g] = (w_g, mu_g, C_g)

        """ E-step t+1 """
        logscore_marginals, logscore_joint = logpdf_GMM(D, GMM_params)
        logscore_posterior = logscore_joint - logscore_marginals  # Broadcasting
        gammas = np.exp(logscore_posterior)  # GxN np.array where G is the number of clusters and N is the number of samples

        # Check if the algorithm has converged
        loglikelihood = np.average(logscore_marginals)

        # print("ll at step " + str(t) + ": " + str(loglikelihood))
        if loglikelihood < last_loglikelihood:
            print(f"!!WARNING!! ll at step {t} decreased")

        if abs(loglikelihood - last_loglikelihood) < delta:
            good_enough = True
        last_loglikelihood = loglikelihood
        t += 1

    return GMM_params


def LBG(D, nof_iterations, delta=1e-6, min_cov=0.01, alpha=0.1, variant="std", verbose=False):
    """ Returns all the optimized parameters of the GMM from iteration 0 to nof_iterations.
        This means that the returned list has a total of (nof_iterations + 1) sets of parameters """
    if variant not in ["std", "diag", "tied"]:
        raise ValueError("The variant must be either 'std', 'diag' or 'tied'")

    all_params = []

    """ Initialize the GMM using MVG MLs """
    # First of all the mean
    mu = np.mean(D, axis=1)

    # And then the Covariance Matrix
    C = get_covariance_matrix(D)
    if variant == "diag":
        C = C * np.eye(C.shape[0])

    # Enforce the minimum covariance constraint
    U, s, _ = np.linalg.svd(C)
    s[s < min_cov] = min_cov
    C = np.dot(U, vcol(s) * U.T)

    GMM_params = [(1.0, mu, C)]

    all_params.append(GMM_params)

    # Split the components
    for _ in tqdm(range(0, nof_iterations), desc="LBG iterations..."):
        new_GMM_params = []
        for (w_g, mu_g, C_g) in GMM_params:
            # Calculating the displacement vector d as the first eigenvector of the covariance matrix
            U, s, Vh = np.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            d = d.ravel()

            new_GMM_params.append((0.5 * w_g, mu_g - d, C_g))
            new_GMM_params.append((0.5 * w_g, mu_g + d, C_g))

        GMM_params = EMM(D, new_GMM_params, delta=delta, min_cov=min_cov, variant=variant)
        all_params.append(GMM_params)

    return all_params
