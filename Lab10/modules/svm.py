from typing import Tuple
import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from modules.common_matrix_operations import vcol, vrow
from modules.evaluation import get_dcf__includes_classification, get_min_normalized_dcf_binary, \
    get_normalized_dcf_binary
from modules.statistics import how_many_samples, how_many_features\


def get_H(D: ndarray, z: ndarray) -> ndarray:
    if z.ndim != 1:
        raise ValueError('The labels must be a 1D array')
    if z.shape[0] != D.shape[1]:
        raise ValueError('The number of labels must be equal to the number of samples')

    G = D.T @ D
    zs = vcol(z) @ vcol(z).T
    H = zs * G

    return H


def get_H_with_loops(D_tr, z):
    H = np.zeros((D_tr.shape[1], D_tr.shape[1]))
    for i in range(D_tr.shape[1]):
        for j in range(D_tr.shape[1]):
            H[i, j] = z[i] * z[j] * D_tr[:, i].T @ D_tr[:, j]

    return H


def get_kern_H_with_loops(D_tr, z, ker, ker_params):
    H = np.zeros((D_tr.shape[1], D_tr.shape[1]))
    for i in tqdm(range(D_tr.shape[1]), desc="Loading..."):
        for j in range(D_tr.shape[1]):
            H[i, j] = z[i] * z[j] * ker(D_tr[:, i], D_tr[:, j], *ker_params)

    print()
    return H


def get_kern_H(D_tr, z, ker, ker_params):
    # Compute the kernel matrix
    K = np.array([[ker(D_tr[:, i], D_tr[:, j], *ker_params)
                   for j in range(D_tr.shape[1])]
                  for i in range(D_tr.shape[1])])

    # Compute the outer product of z
    zs = vcol(z) @ vcol(z).T

    # Multiply the kernel matrix by the outer product of z
    H = zs * K

    return H


def dualJ_with_gradient(alphas: ndarray, H: ndarray) -> Tuple[float, np.ndarray]:
    dualJ = 0.5 * vcol(alphas).T @ H @ vcol(alphas) - np.sum(alphas)
    gradient = H @ vcol(alphas) - 1

    return dualJ.item(), gradient.ravel()


def primalJ(w: ndarray, D_tr, L_tr, C) -> Tuple[float, np.ndarray]:
    w = vcol(w)
    z = 2 * L_tr - 1
    regularization_term = 0.5 * w.T @ w

    opp_scores = 1 - z * (w.T @ D_tr)
    maximums = np.maximum(opp_scores, 0)
    loss_term = C * np.sum(maximums)

    return regularization_term + loss_term


def poly_kernel(x: ndarray, y: ndarray, c: int, d: int, K: float) -> float:
    """ The bias term ξ is to be passed through its square root K """
    return ((vrow(x) @ vcol(y) + c) ** d).item() + K**2


def rbf_kernel(x: ndarray, y: ndarray, gamma: float, K: float) -> float:
    """ The bias term ξ is to be passed through its square root K """
    exp_term = -gamma * (np.linalg.norm(x.ravel() - y.ravel()) ** 2)
    return np.exp(exp_term).item() + K**2


def run_dual(D_tr_expanded: ndarray, L_tr: ndarray, D_val_expanded: ndarray, L_val: ndarray, C: float, pi=0.5) -> Tuple[float, float, float, float]:
    """ Remember to add the K row to the Data for this to work """
    z = 2 * L_tr - 1

    # x0 should have as many values as there are samples
    x0 = np.zeros(how_many_samples(D_tr_expanded))
    # We specify the bounds for all α to be between 0 and C
    bounds = [(0, C) for _ in x0]

    # Training, i.e. finding the optimal α and converting it to optimal w
    H = get_H(D_tr_expanded, z)
    alphas, loss, _ = scipy.optimize.fmin_l_bfgs_b(func=dualJ_with_gradient, x0=x0, args=(H,), bounds=bounds, approx_grad=False, factr=1.0)
    w = np.sum(vrow(alphas * z) * D_tr_expanded, axis=1)
    w = vcol(w)

    # Validation
    scores = w.T @ D_val_expanded
    predicted_labels = np.sign(scores)
    predicted_labels[predicted_labels == -1] = 0

    # Error rate and DCFs
    error_rate = 1 - np.mean(predicted_labels == L_val)

    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(scores, L_val, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(scores, L_val, *triplet)

    return -loss, error_rate, dcf_min, dcf_norm


def run_primal(D_tr_expanded: ndarray, L_tr: ndarray, D_val_expanded: ndarray, L_val: ndarray, C: float, pi=0.5) -> Tuple[float, float, float, float]:
    """ Remember to add the K row to the Data for this to work """
    # x0 should have as many values as there are samples
    x0 = np.zeros(how_many_features(D_tr_expanded)) + 1

    # Training, i.e. finding the optimal α and converting it to optimal w
    w, loss, _ = scipy.optimize.fmin_l_bfgs_b(func=primalJ, x0=x0, args=(D_tr_expanded, L_tr, C), approx_grad=True, factr=1.0)
    w = vcol(w)

    # Validation
    scores = w.T @ D_val_expanded
    predicted_labels = np.sign(scores)
    predicted_labels[predicted_labels == -1] = 0

    # Error rate and DCFs
    error_rate = 1 - np.mean(predicted_labels == L_val)

    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(scores, L_val, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(scores, L_val, *triplet)

    return loss, error_rate, dcf_min, dcf_norm


def run_dual_with_kernel(D_tr: ndarray, L_tr: ndarray, D_val: ndarray, L_val: ndarray, C: float,
                         H, score_kernel_matrix, pi=0.5) -> Tuple[float, float, float, float]:
    z = 2 * L_tr - 1

    # x0 should have as many avlues as there are samples
    x0 = np.zeros(how_many_samples(D_tr))
    # We specify the bounds for all α to be between 0 and C
    bounds = [(0, C) for _ in x0]

    # Training, i.e. finding the optimal α and converting it to optimal w
    alphas, loss, _ = scipy.optimize.fmin_l_bfgs_b(func=dualJ_with_gradient, x0=x0, args=(H,), bounds=bounds, approx_grad=False, factr=1.0)

    print()
    print("Computing Scores:")
    # Validation
    scores = np.zeros(how_many_samples(D_val))
    for j in tqdm(range(0, how_many_samples(D_val)), desc="Loading..."):
        for i in range(0, how_many_samples(D_tr)):
            if alphas[i] == 0:
                continue
            scores[j] += alphas[i] * z[i] * score_kernel_matrix[i, j]

    predicted_labels = np.sign(scores)
    predicted_labels[predicted_labels == -1] = 0

    # Error rate and DCFs
    error_rate = 1 - np.mean(predicted_labels == L_val)

    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(scores, L_val, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(scores, L_val, *triplet)

    return -loss, error_rate, dcf_min, dcf_norm


def expand_data_with_bias_regterm(D, K):
    return np.vstack((D, K * np.ones(D.shape[1])))


def plot_dcfs(Cs, act_dcfs, min_dcfs, plot_name):
    plt.figure()
    plt.title(plot_name)
    plt.xlabel('C')
    plt.ylabel('DCFs')
    plt.xscale('log', base=10)
    plt.plot(Cs, act_dcfs, label='actDCF')
    plt.plot(Cs, min_dcfs, label='minDCF')
    plt.legend()

    plt.savefig(plot_name)
    print(f"[[Saved {plot_name}]]")
    plt.show()
