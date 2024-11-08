import numpy as np
import scipy


from modules.common_matrix_operations import vrow, vcol
from modules.evaluation import get_dcf__includes_classification, get_normalized_dcf_binary, \
    get_min_normalized_dcf_binary
from modules.load_datasets import split_db_2to1, load_iris_binary
from modules.logistic_regression import logreg_obj_prior_weighted, logreg_obj
from modules.statistics import get_empirical_prior_binary


def logreg_non_weighted(DTR, LTR, DVAL, LVAL, pi):
    pi_emp = get_empirical_prior_binary(LTR)
    print(f"Non-weighted Logistic Regression")

    x0 = np.zeros(DTR.shape[0] + 1)

    ls = [0.001, 0.1, 1.0]
    print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'emp_π = ' + str(pi_emp):<{10}}")
    for l in ls:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, args=(DTR, LTR, l), approx_grad=False)

        w = x[:-1]
        b = x[-1]

        logposteriors = (vcol(w).T @ DVAL + b).ravel()
        predicted_labels = np.sign(logposteriors)
        predicted_labels[predicted_labels == -1] = 0

        llrs = logposteriors - np.log(pi_emp / (1 - pi_emp))

        error_rate = 1 - np.mean(predicted_labels == LVAL)

        triplet = (pi, 1, 1)
        dcf, _, _ = get_dcf__includes_classification(llrs, LVAL, *triplet)
        dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
        dcf_min = get_min_normalized_dcf_binary(llrs, LVAL, *triplet)

        print(f"{l:.0e}     {f:.6e}   {error_rate*100:.2f}%    {dcf_min:.4f}    {dcf_norm:.4f}")


def logreg_weighted(DTR, LTR, DVAL, LVAL, pi):
    print(f"Prior-weighted Logistic Regression")

    x0 = np.zeros(DTR.shape[0] + 1)

    ls = [0.001, 0.1, 1.0]
    print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'π = ' + str(pi):<{10}}")
    for l in ls:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj_prior_weighted, x0=x0, args=(DTR, LTR, l, pi), approx_grad=False)

        w = x[:-1]
        b = x[-1]

        logposteriors = (vcol(w).T @ DVAL + b).ravel()
        predicted_labels = np.sign(logposteriors)
        predicted_labels[predicted_labels == -1] = 0

        llrs = logposteriors - np.log(pi / (1 - pi))

        error_rate = 1 - np.mean(predicted_labels == LVAL)

        triplet = (pi, 1, 1)
        dcf, _, _ = get_dcf__includes_classification(llrs, LVAL, *triplet)
        dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
        dcf_min = get_min_normalized_dcf_binary(llrs, LVAL, *triplet)

        print(f"{l:.0e}     {f:.6e}   {error_rate*100:.2f}%    {dcf_min:.4f}    {dcf_norm:.4f}")


if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # print_dataset_info(DTR, LTR, "Training")
    # print_dataset_info(DVAL, LVAL, "Validation")
    # print(get_unique_classes(LTR), get_unique_classes(LVAL))

    logreg_non_weighted(DTR, LTR, DVAL, LVAL)

    pi_emp = get_empirical_prior_binary(LTR)
    logreg_weighted(DTR, LTR, DVAL, LVAL, pi_emp)
    logreg_weighted(DTR, LTR, DVAL, LVAL, 0.8)





