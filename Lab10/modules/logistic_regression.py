import numpy as np
import scipy
from typing import Tuple

from modules.common_matrix_operations import vrow, vcol
from modules.evaluation import get_dcf__includes_classification, get_min_normalized_dcf_binary, \
    get_normalized_dcf_binary
from modules.statistics import get_empirical_prior_binary, how_many_features


def logreg_obj(v, DTR, LTR, l):
    # Extract parameters w and b from vector v
    w = v[:-1]
    b = v[-1]

    # Compute z_i for each label in LTR
    ZTR = 2 * LTR - 1

    # Compute the regularization term
    regularization_term = (l / 2) * np.sum(w ** 2)

    # Compute the loss term
    scores = (np.dot(w, DTR) + b).ravel()
    loss_term = np.mean(np.logaddexp(0, -ZTR * scores))

    # Combine both terms to get the objective function value
    J = regularization_term + loss_term

    # Compute the gradient
    G = -ZTR / (1.0 + np.exp(ZTR * scores))
    J_w = np.mean(vrow(G) * DTR, axis=1) + l * w
    J_b = np.mean(G)

    return J, np.hstack((J_w, J_b))


def logreg_obj_prior_weighted(v, DTR, LTR, l, pi):
    # Calucate n_t and n_f
    n_t = np.sum(LTR)
    n_f = len(LTR) - n_t

    # Extract parameters w and b from vector v
    w = v[:-1]
    b = v[-1]

    # Compute z_i for each label in LTR. Also compute ξ (weight)
    ZTR = 2 * LTR - 1
    weight = [pi / n_t if x == 1 else (1 - pi) / n_f for x in ZTR]

    # Compute the regularization term
    regularization_term = (l / 2) * np.sum(w ** 2)

    # Compute the loss term
    scores = (np.dot(w, DTR) + b).ravel()
    summation_terms = weight * np.logaddexp(0, -ZTR * scores)
    loss_term = np.sum(summation_terms)

    # Combine both terms to get the objective function value
    J = regularization_term + loss_term

    # Compute the gradient
    G = -ZTR / (1.0 + np.exp(ZTR * scores))
    J_w = np.sum(weight * vrow(G) * DTR, axis=1) + l * w
    J_b = np.sum(weight * G)

    return J, np.hstack((J_w, J_b))


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


def classify_binary_logreg(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray,
                           l: float, pi: float, verbose: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
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
    x0 = np.zeros(DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, args=(DTR, LTR, l), approx_grad=False)
    w = x[:-1]  # Where: score = w.T @ x + b
    b = x[-1]

    # Compute the log-posteriors and the predicted labels
    logposteriors = (vcol(w).T @ DVAL + b).ravel()
    predicted_labels = np.sign(logposteriors)
    predicted_labels[predicted_labels == -1] = 0
    # Extract the LLRs
    llrs = logposteriors - np.log(pi_emp / (1 - pi_emp))

    # Compute the error rate
    error_rate = 1 - np.mean(predicted_labels == LVAL)
    # Compute the DCFs
    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(llrs, LVAL, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(llrs, LVAL, *triplet)

    if verbose:
        # print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'emp_π = ' + str(pi_emp):<{10}}")
        print(f"{l:.0e}     {f:.6e}   {error_rate * 100:.2f}%    {dcf_min:.4f}    {dcf_norm:.4f}")

    # Return the log-posteriors, the error rate, and the DCFs
    return (logposteriors, llrs), error_rate, (dcf_norm, dcf_min)


def classify_binary_logreg_prior_weighted(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray,
                                    l: float, pi: float, verbose: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
    """
    Classify the validation set using binary logistic regression with prior-weighted.
    """
    # Check if the labels are binary
    unique_labels = np.unique(np.concatenate((LTR, LVAL)))
    if len(unique_labels) != 2:
        raise ValueError("Labels must be binary for binary logistic regression.")

    # Optimize the objective function
    x0 = np.zeros(DTR.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj_prior_weighted, x0=x0, args=(DTR, LTR, l, pi), approx_grad=False)
    w = x[:-1]  # Where: score = w.T @ x + b
    b = x[-1]

    # Compute the log-posteriors and the predicted labels
    logposteriors = (vcol(w).T @ DVAL + b).ravel()
    predicted_labels = np.sign(logposteriors)
    predicted_labels[predicted_labels == -1] = 0
    # Extract the LLRs
    llrs = logposteriors - np.log(pi / (1 - pi))

    # Compute the error rate
    error_rate = 1 - np.mean(predicted_labels == LVAL)
    # Compute the DCFs
    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(llrs, LVAL, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(llrs, LVAL, *triplet)

    if verbose:
        # print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'emp_π = ' + str(pi_emp):<{10}}")
        print(f"{l:.0e}     {f:.6e}   {error_rate * 100:.2f}%    {dcf_min:.4f}    {dcf_norm:.4f}")

    # Return the log-posteriors, the error rate, and the DCFs
    return (logposteriors, llrs), error_rate, (dcf_norm, dcf_min)


def classify_binary_logreg_quadratic(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray,
                                     l: float, pi: float, verbose: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
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

    # Compute the error rate
    error_rate = 1 - np.mean(predicted_labels == LVAL)
    # Compute the DCFs
    triplet = (pi, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
    dcf, _, _ = get_dcf__includes_classification(llrs, LVAL, *triplet)
    dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
    dcf_min = get_min_normalized_dcf_binary(llrs, LVAL, *triplet)

    if verbose:
        # print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'emp_π = ' + str(pi_emp):<{10}}")
        print(f"{l:.0e}     {f:.6e}   {error_rate * 100:.2f}%    {dcf_min:.4f}    {dcf_norm:.4f}")

    # Return the log-posteriors, the error rate, and the DCFs
    return (logposteriors, llrs), error_rate, (dcf_norm, dcf_min)


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
