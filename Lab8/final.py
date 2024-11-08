import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from modules.best_classifiers_clean_version import binary_logreg_quadratic__train
from modules.common_matrix_operations import vcol
from modules.load_datasets import load_fingerprints, split_db_2to1, shrink_dataset, load_eval_data
from modules.logistic_regression import classify_binary_logreg, classify_binary_logreg_prior_weighted, \
    classify_binary_logreg_quadratic
from modules.statistics import get_mean


# Suppress displaying plots
def nop():
    pass
plt.show = nop


def plot_dcfs(lambdas, act_dcfs, min_dcfs, plot_name):
    plt.figure()
    plt.title(plot_name)
    plt.xlabel('λ')
    plt.ylabel('DCFs')
    plt.xscale('log', base=10)
    plt.plot(lambdas, act_dcfs, label='actDCF')
    plt.plot(lambdas, min_dcfs, label='minDCF')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'output_first/{plot_name}.png')
    print(f"[[Saved {plot_name}.png]]")
    plt.clf()


def eval_nonweigted_logreg(DTR, LTR, DVAL, LVAL, verbose=False):
    act_dcfs = []
    min_dcfs = []

    pi = 0.1
    lambdas = np.logspace(-4, 2, 13)
    # print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'π = ' + str(pi):<{10}}")
    for l in lambdas:
        (_, _), _, (dcf_norm, dcf_min) = classify_binary_logreg(DTR, LTR, DVAL, LVAL, l, pi, verbose=verbose)

        act_dcfs.append(dcf_norm)
        min_dcfs.append(dcf_min)

    plot_dcfs(lambdas, act_dcfs, min_dcfs, "Fingerprints - Non-weighted Logistic Regression - pi=0.1")


def eval_prior_weighted_logreg(DTR, LTR, DVAL, LVAL, verbose=False):
    act_dcfs = []
    min_dcfs = []

    pi = 0.1
    lambdas = np.logspace(-4, 2, 13)
    if verbose:
        print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'π = ' + str(pi):<{10}}")
    for l in lambdas:
        (_, _), _, (dcf_norm, dcf_min) = classify_binary_logreg_prior_weighted(DTR, LTR, DVAL, LVAL, l, pi, verbose=verbose)

        act_dcfs.append(dcf_norm)
        min_dcfs.append(dcf_min)

    plot_dcfs(lambdas, act_dcfs, min_dcfs, "Fingerprints - Prior-Weighted Logistic Regression - pi=0.1")


def eval_quadratic_logreg(DTR, LTR, DVAL, LVAL, verbose=False):
    act_dcfs = []
    min_dcfs = []

    pi = 0.1
    lambdas = np.logspace(-4, 2, 13)
    if verbose:
        print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'π = ' + str(pi):<{10}}")
    for l in lambdas:
        (logposteriors, llrs), error_rate, (dcf_norm, dcf_min) = classify_binary_logreg_quadratic(DTR, LTR, DVAL, LVAL, l, pi, verbose=verbose)

        act_dcfs.append(dcf_norm)
        min_dcfs.append(dcf_min)

        # For Lab10 checking
        np.save(f'llrs_LR_l{l}.npy', llrs)
        print(f"[[Saved ./llrs_LR_l{l}.npy]]")

    plot_dcfs(lambdas, act_dcfs, min_dcfs, "Fingerprints - Non-Weighted Quadratic Logistic Regression - pi=0.1")


def eval_nonweigted_precentering(DTR, LTR, DVAL, LVAL, verbose=False):
    dataset_mean = get_mean(DTR)
    DTRc = DTR - vcol(dataset_mean)
    DVALc = DVAL - vcol(dataset_mean)

    act_dcfs = []
    min_dcfs = []

    pi = 0.1
    lambdas = np.logspace(-4, 2, 13)
    if verbose:
        print(f"{'λ':<10}{'J':<15}{'err':<10}{'minDCF':<10}{'actDCF':<10}{'π = ' + str(pi):<{10}}")
    for l in lambdas:
        (_, _), _, (dcf_norm, dcf_min) = classify_binary_logreg(DTRc, LTR, DVALc, LVAL, l, pi, verbose=verbose)

        act_dcfs.append(dcf_norm)
        min_dcfs.append(dcf_min)

    plot_dcfs(lambdas, act_dcfs, min_dcfs, "Fingerprints - Non-weighted Logistic Regression Precentering - pi=0.1")


if __name__ == '__main__':
    D, L = load_fingerprints()
    # D, L = shrink_dataset(D, L, 10)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    evalD, evalL = load_eval_data()

    # print("============================== Non Weighted ==============================")
    # # Plots the DCFs for the non-weighted logistic regression. We use the full dataset here
    # eval_nonweigted_logreg(DTR, LTR, DVAL, LVAL, verbose=True)

    # print("========================== Non Weighted Small Set =========================")
    # # Now we use a smaller set instead (1/50 of the training set)
    # eval_nonweigted_logreg(DTR[:, ::50], LTR[::50], DVAL, LVAL, verbose=True)

    print("============================= Prior Weighted =============================")
    # Plots DCFs for the prior-weighted logistic regression.
    eval_prior_weighted_logreg(DTR, LTR, evalD, evalL, verbose=True)

    print("======================== Quadratic (Non Weighted) ========================")
    # Plots DCFs for the quadratic logistic regression.
    eval_quadratic_logreg(DTR, LTR, evalD, evalL, verbose=True)

    print("===================== Precentering (Non Weighted) ========================")
    # Getting back to the linear non-weighted model, with centering preprocessing
    eval_nonweigted_precentering(DTR, LTR, evalD, evalL, verbose=True)

    # print("===================== Quadratic for final! ========================")
    # binary_logreg_quadratic__train(DTR, LTR, l=0.0031)
