import pickle
from typing import Tuple, Callable

import numpy as np
import scipy
from numpy import ndarray
from tqdm import tqdm

from modules.common_matrix_operations import vrow, vcol, to_numpy_matrix
from modules.probability_first import logpdf_GAU_ND
from modules.statistics import get_empirical_prior_binary, how_many_samples


""" =========== Logistic Regression =========== """


class LogisticRegressionParams:
    w: ndarray
    b: float
    pi: float
    variant: str

    def __init__(self, w: ndarray, b: float, pi: float, variant: str):
        self.w = w
        self.b = b
        self.pi = pi
        self.variant = variant

    def __str__(self) -> str:
        return f"w: {self.w}, b: {self.b}, pi: {self.pi}"

    def get_wb(self) -> Tuple[ndarray, float]:
        return self.w, self.b

    def get_pi(self) -> float:
        """ Returns the empirical prior if the model is non-weighted, otherwise the prior used when training the
        weighted model"""
        return self.pi

    def get_variant(self) -> str:
        """ Returns the variant of the model, i.e. either 'nonweighted' or 'priorweighted' or 'quadratic (
        nonweighted)'"""
        return self.variant

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance


def logreg_quadratic_obj(v, PHI_DTR, LTR, l):
    # Extract parameters w and c from vector v
    w = v[:-1]
    c = v[-1]

    # Compute z_i for each label in LTR
    ZTR = 2 * LTR - 1

    # Compute the regularization term
    regularization_term = (l / 2) * np.sum(w ** 2)

    # Compute the loss term
    scores = (np.dot(w, PHI_DTR) + c).ravel()
    loss_term = np.mean(np.logaddexp(0, -ZTR * scores))

    # Combine both terms to get the objective function value
    J = regularization_term + loss_term

    # Compute the gradient
    G = -ZTR / (1.0 + np.exp(ZTR * scores))
    J_w = np.mean(vrow(G) * PHI_DTR, axis=1) + l * w
    J_b = np.mean(G)

    return J, np.hstack((J_w, J_b))


def get_phi_of_x(D):
    n, m = D.shape
    phi = np.empty(((n * n) + n, m))

    # Compute vec(x@x.T) for each column in D
    for i in range(m):
        x = D[:, i].reshape(-1, 1)  # Get the i-th vector and reshape to (n, 1)
        xxT = x @ x.T  # Compute the outer product, resulting in (n, n)
        vec_xxT = xxT.ravel(order='F')
        phi[:, i] = np.concatenate((vec_xxT, x.ravel()))  # Flatten and store in the corresponding column

    return phi


def binary_logreg_quadratic__train(DTR: np.ndarray, LTR: np.ndarray, l: float) -> Tuple[np.ndarray, float, float]:
    """  Train a binary logistic regression model with non-weighted. """
    if DTR.ndim != 2:
        raise ValueError(f"DTR matrix should be nxm, instead is {DTR.shape}")
    if LTR.ndim != 1:
        raise ValueError(f"LTR matrix should be 1xn, instead is {LTR.shape}")
    # Check if the labels are binary
    unique_labels = np.unique(LTR)
    if len(unique_labels) != 2:
        raise ValueError("Labels must be binary for binary logistic regression.")

    # Compute the empirical prior
    pi_emp = get_empirical_prior_binary(LTR)

    # Optimize the objective function
    PHI_DTR = get_phi_of_x(DTR)
    x0 = np.zeros(PHI_DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_quadratic_obj, x0=x0, args=(PHI_DTR, LTR, l), approx_grad=False)
    w = x[:-1]  # Where: score = w.T @ x + b
    c = x[-1] - np.log(pi_emp / (1 - pi_emp))

    LRmodel = LogisticRegressionParams(w, c, pi_emp, "quadratic")
    LRmodel.save_to_file(f"output_first/model_qLR.pkl")
    print(f"[[Saved model_qLR.pkl]]")

    return w, c, pi_emp


def binary_logreg_quadratic__classify(w, b, DVAL: np.ndarray, pi_emp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if DVAL.ndim != 2:
        raise ValueError(f"DVAL matrix should be nxm, instead is {DVAL.shape}")

    # Compute the log-posteriors and the predicted labels
    PHI_DVAL = get_phi_of_x(DVAL)
    llrs = (vcol(w).T @ PHI_DVAL + b).ravel()

    logposteriors = llrs + np.log(pi_emp / (1 - pi_emp))
    predicted_labels = np.sign(logposteriors)
    predicted_labels[predicted_labels == -1] = 0

    return llrs, logposteriors, predicted_labels


""" =========== Support Vector Machines =========== """


class SVMParameters:
    alphas: ndarray
    SVs: ndarray
    z: ndarray
    variant: str

    def __init__(self, alphas: ndarray, SVs: ndarray, z: ndarray, variant: str):
        self.alphas = alphas
        self.SVs = SVs
        self.variant = variant
        self.z = z

    def get_params(self) -> Tuple[ndarray, ndarray, ndarray]:
        return self.alphas, self.SVs, self.z

    def get_variant(self) -> str:
        """ Returns the variant of the model, i.e. either 'linear' or 'pkern' or 'rbfkern """
        return self.variant

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance


def dualJ_with_gradient(alphas: ndarray, H: ndarray) -> Tuple[float, np.ndarray]:
    dualJ = 0.5 * vcol(alphas).T @ H @ vcol(alphas) - np.sum(alphas)
    gradient = H @ vcol(alphas) - 1

    return dualJ.item(), gradient.ravel()


def run_dual_with_kernel__only_train(D_tr: ndarray, L_tr: ndarray, C: float, H) -> SVMParameters:
    # x0 should have as many avlues as there are samples
    x0 = np.zeros(how_many_samples(D_tr))
    # We specify the bounds for all α to be between 0 and C
    bounds = [(0, C) for _ in x0]

    # Training, i.e. finding the optimal α and converting it to optimal w
    alphas, loss, _ = scipy.optimize.fmin_l_bfgs_b(func=dualJ_with_gradient, x0=x0, args=(H,), bounds=bounds,
                                                   approx_grad=False, factr=1.0)

    z = 2 * L_tr - 1

    alphasSV = alphas[alphas > 1e-5]
    zSV = z[alphas > 1e-5]
    support_vectors = D_tr[:, alphas > 1e-5]
    SVMmodel = SVMParameters(alphas=alphasSV, SVs=support_vectors, z=zSV, variant='rbfkern')
    SVMmodel.save_to_file(f"output_first/model_rbfSVM.pkl")
    print(f"[[Saved model_rbfSVM.pkl]]")

    return SVMmodel


def run_dual_with_kernel__only_classification(SVMmodel: SVMParameters, D_val: ndarray, kernel: Callable,
                                              kernel_params: Tuple) -> Tuple[ndarray, ndarray]:
    alphas, support_vectors, z = SVMmodel.get_params()
    scores = np.zeros(how_many_samples(D_val))
    for j in tqdm(range(how_many_samples(D_val)), desc="Loading..."):
        for i in range(alphas.size):
            scores[j] += alphas[i] * z[i] * kernel(support_vectors[:, i], D_val[:, j], *kernel_params)

    predicted_labels = np.sign(scores)
    predicted_labels[predicted_labels == -1] = 0
    np.save(f"output_first/scores_rbfSVM.npy", scores)
    print(f"[[Saved scores_rbfSVM.npy]]")

    return scores, predicted_labels


""" =========== Gaussian Mixture Models =========== """


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


def classsify_GMM(D: ndarray, GMM_class0_model: GMMParameters, GMM_class1_model: GMMParameters):
    class0_params = GMM_class0_model.get_components()
    class1_params = GMM_class1_model.get_components()

    # 1) Log-likelihoods class by class
    loglikelihoods_of_class_0, _ = logpdf_GMM(D, class0_params)
    loglikelihoods_of_class_1, _ = logpdf_GMM(D, class1_params)
    loglikelihoods_by_class = np.vstack((loglikelihoods_of_class_0, loglikelihoods_of_class_1))

    # 2) Compute the log joint likelihoods, again class by class
    priors = np.array([0.5, 0.5])
    log_joints_by_class = loglikelihoods_by_class
    for i in range(len(priors)):
        log_joints_by_class[i, :] = loglikelihoods_by_class[i, :] + np.log(priors[i])

    llrs = log_joints_by_class[1, :] - log_joints_by_class[0, :]
    return llrs
