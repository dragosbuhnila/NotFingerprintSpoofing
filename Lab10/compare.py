from typing import Tuple

import numpy as np
import scipy
from numpy import ndarray
from tqdm import tqdm

from modules.GMM import LBG, logpdf_GMM
from modules.common_matrix_operations import vcol
from modules.load_datasets import load_fingerprints, split_db_2to1, shrink_dataset
from modules.logistic_regression import get_phi_of_x, logreg_quadratic_obj
from modules.plottings import plot_bayes_error_plots
from modules.statistics import how_many_samples, get_empirical_prior_binary, how_many_classes
from modules.svm import dualJ_with_gradient, get_kern_H_with_loops


def rbf_kernel(x: ndarray, y: ndarray, gamma: float, K: float) -> float:
    """ The bias term ξ is to be passed through its square root K """
    exp_term = -gamma * (np.linalg.norm(x.ravel() - y.ravel()) ** 2)
    return np.exp(exp_term).item() + K**2


def get_kernelSVM_llrs(D_tr: ndarray, L_tr: ndarray, D_val: ndarray, L_val: ndarray, C: float,
                         z, H, score_kernel_matrix) -> ndarray:
    # x0 should have as many values as there are samples
    x0 = np.zeros(how_many_samples(D_tr))
    # We specify the bounds for all α to be between 0 and C
    bounds = [(0, C) for _ in x0]

    # Training, i.e. finding the optimal α
    alphas, _, _ = scipy.optimize.fmin_l_bfgs_b(func=dualJ_with_gradient, x0=x0, args=(H,), bounds=bounds, approx_grad=False, factr=1.0)

    # Validation
    scores = np.zeros(how_many_samples(D_val))
    for j in tqdm(range(0, how_many_samples(D_val)), desc="Loading Scores..."):
        for i in range(0, how_many_samples(D_tr)):
            if alphas[i] == 0:
                continue
            scores[j] += alphas[i] * z[i] * score_kernel_matrix[i, j]

    # scores and llrs are really the same thing
    return scores


def get_QuadraticLR_llrs(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray, l: float) -> ndarray:
    """
    Classify the validation set using binary logistic regression with non-weighted.
    Note that pi is the prior probability used JUST in the DCF calculation. It is not used in the optimization process and for extracting llrs from logposteriors.
    """
    # Check if the labels are binary
    unique_labels = np.unique(np.concatenate((LTR, LVAL)))
    if len(unique_labels) != 2:
        raise ValueError("Labels must be binary for binary logistic regression.")

    # Compute the empirical prior
    pi_emp = get_empirical_prior_binary(LTR)

    # Optimize the objective function
    PHI_DTR = get_phi_of_x(DTR)
    x0 = np.zeros(PHI_DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_quadratic_obj, x0=x0, args=(PHI_DTR, LTR, l), approx_grad=False)
    w = x[:-1]  # Where: score = w.T @ x + b
    c = x[-1]

    # Compute the log-posteriors and the predicted labels
    PHI_DVAL = get_phi_of_x(DVAL)
    logposteriors = (vcol(w).T @ PHI_DVAL + c).ravel()
    predicted_labels = np.sign(logposteriors)
    predicted_labels[predicted_labels == -1] = 0
    # Extract the LLRs
    llrs = logposteriors - np.log(pi_emp / (1 - pi_emp))

    return llrs


def get_GMM_llrs(D_tr, L_tr, D_val, L_val, total_LBG_iterations, variant="diag") -> ndarray:
    all_params_byclass = {"std": [],
                          "diag": [],
                          "tied": []}

    # Training
    for c in tqdm(range(how_many_classes(L)), desc=f"Training {variant} GMMs..."):
        all_params_byclass[variant].append( LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant=variant))

    DCFs = [variant]
    lbg_num = 4
    # 1) Compute log-likelihoods class by class
    loglikelihoods_by_class = np.zeros((how_many_classes(L), how_many_samples(D_val)))
    for (params_by_class, class_x) in zip(all_params_byclass[variant], range(how_many_classes(L))):
        loglikelihoods_of_class_x, _ = logpdf_GMM(D_val, params_by_class[lbg_num])
        loglikelihoods_by_class[class_x, :] = loglikelihoods_of_class_x

    # 2) Compute the log joint likelihoods, again class by class
    priors = np.array([1 / how_many_classes(L) for _ in range(how_many_classes(L))])
    log_joints_by_class = loglikelihoods_by_class
    for i in range(len(priors)):
        log_joints_by_class[i, :] = loglikelihoods_by_class[i, :] + np.log(priors[i])

    llrs = log_joints_by_class[1, :] - log_joints_by_class[0, :]
    return llrs


def save_QuadraticLR_llrs(D_tr, L_tr, D_val, L_val):
    """ === Logistic Regression - l=0.0031 === """
    llrs_LR = get_QuadraticLR_llrs(D_tr, L_tr, D_val, L_val, 0.0031)
    np.save('llrs_LR.npy', llrs_LR)

    print("[[Saved ./llrs_LR.npy]]")
    return


def save_GMM_llrs(D_tr, L_tr, D_val, L_val):
    """ === GMM - 16 clusters (i.e. 4 iterations), and diagonal === """
    llrs_GMM = get_GMM_llrs(D_tr, L_tr, D_val, L_val, 4, "diag")
    np.save('llrs_GMM.npy', llrs_GMM)

    print("[[Saved ./llrs_GMM.npy]]")
    return


def save_kernelSVM_llrs(D_tr, L_tr, D_val, L_val):
    """ === SVM - RBF kernel with γ=np.exp(-2), and C = 31 === """
    # Various necessary preparatory steps
    rbf_kernel_params = (np.exp(-2), 1)
    z = 2 * L_tr - 1

    H = get_kern_H_with_loops(D_tr, z, rbf_kernel, rbf_kernel_params)

    score_kernel_matrix = np.zeros((how_many_samples(D_tr), how_many_samples(D_val)))
    for j in tqdm(range(0, how_many_samples(D_val)), desc="Loading score_kernel_matrix..."):
        for i in range(0, how_many_samples(D_tr)):
            score_kernel_matrix[i, j] = rbf_kernel(D_tr[:, i], D_val[:, j], *rbf_kernel_params)

    llrs_SVM = get_kernelSVM_llrs(D_tr, L_tr, D_val, L_val, 31, z, H, score_kernel_matrix)
    np.save('llrs_SVM.npy', llrs_SVM)

    print("[[Saved ./llrs_SVM.npy]]")
    return


def plot_bayes_error(L_val):
    llrs_SVM = np.load('llrs/llrs_SVM.npy')
    llrs_LR = np.load('llrs/llrs_LR.npy')
    llrs_GMM = np.load('llrs/llrs_GMM.npy')

    plot_bayes_error_plots([llrs_SVM, llrs_LR, llrs_GMM], L_val, ["RBF_SVM", "QLR", "16C_GMM"], "SVM_vs_QLR_vs_GMM")


if __name__ == '__main__':
    """ Load and split the dataset """
    D, L = load_fingerprints()
    # D, L = shrink_dataset(D, L, 10)
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)

    # save_kernelSVM_llrs(D_tr, L_tr, D_val, L_val)
    # save_QuadraticLR_llrs(D_tr, L_tr, D_val, L_val)
    # save_GMM_llrs(D_tr, L_tr, D_val, L_val)

    # Plot bayes error
    plot_bayes_error(L_val)
