import matplotlib.pyplot as plt
import numpy as np

""" Computed the DCF for a really dumb classifier, which either always labels as
    True or as False, regardless of the data distribution"""
import sys

from modules.classification import get_dcf_binary, get_prior_cost_threshold
from modules.mvg_classification import opt_classify_two_classes
from modules.statistics import get_confusion_matrix, get_fn_fp_rate


def get_reference_dcf_binary(pi, Cfn, Cfp, verbose=False):
    dcf_always_true = pi * Cfn
    dcf_always_false = (1 - pi) * Cfp

    if dcf_always_true <= dcf_always_false:
        if verbose:
            print(f"Reference DCF is {dcf_always_true}")
        return dcf_always_true, "always true"
    else:
        if verbose:
            print(f"Reference DCF is {dcf_always_false}")
        return dcf_always_false, "always false"


def get_normalized_dcf_binary(pi, Cfn, Cfp, dcf):
    ref, _ = get_reference_dcf_binary(pi, Cfn, Cfp)

    normalized_dcf = dcf / ref
    return normalized_dcf


def get_min_normalized_dcf_binary(llrs, labels, pi, Cfn, Cfp, verbose=False, verbose_title=""):
    thresholds = sorted(llrs.tolist())
    thresholds.append(max(thresholds) + 1)
    thresholds.insert(0, min(thresholds) - 1)

    min_dcf = sys.maxsize
    for threshold in thresholds:
        _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
        conf_mat = get_confusion_matrix(predictions, labels,
                                        verbose=verbose, verbose_title=f"{verbose_title} pi={pi}, Cfn={Cfn}, Cfp={Cfp}")

        reference_dcf, _ = get_reference_dcf_binary(pi, Cfn, Cfp)
        cur_dcf = get_dcf_binary(conf_mat, pi, Cfn, Cfp) / reference_dcf
        if cur_dcf < min_dcf:
            min_dcf = cur_dcf

    return min_dcf


def _is_sorted(vector):
    return all(vector[i] <= vector[i + 1] for i in range(len(vector) - 1))


def plot_ROC(llrs, labels, verbose=False, verbose_title=""):
    thresholds = sorted(llrs.tolist())
    thresholds.append(max(thresholds) + 1)
    thresholds.insert(0, min(thresholds) - 1)

    FPRs = []
    TPRs = []

    for threshold in thresholds:
        _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
        conf_mat = get_confusion_matrix(predictions, labels,
                                        verbose=verbose, verbose_title=verbose_title)
        fn_rate, fp_rate = get_fn_fp_rate(conf_mat)
        FPRs.append(fp_rate)
        TPRs.append(1 - fn_rate)

    fpr_tpr_pairs = list(zip(FPRs, TPRs))
    sorted_fpr_tpr_pairs = sorted(fpr_tpr_pairs, key=lambda x: x[0])
    FPRs, TPRs = zip(*sorted_fpr_tpr_pairs)

    auc = np.trapz(TPRs, FPRs)

    plt.figure(num=verbose_title)
    plt.plot(FPRs, TPRs, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(verbose_title)
    plt.legend()
    plt.grid(True)
    plt.show()


def get_effective_prior(pi: float, Cfn: float, Cfp: float) -> float:
    num = pi * Cfn
    den = pi * Cfn + (1 - pi) * Cfp
    eff_prior = num / den

    return eff_prior


def get_dcf__includes_classification(llrs, labels, pi, Cfn, Cfp):
    threshold = get_prior_cost_threshold(pi=pi, Cfn=Cfn, Cfp=Cfp)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels,
                                    verbose=False, verbose_title="")
    dcf = get_dcf_binary(conf_mat, pi=pi, Cfn=Cfn, Cfp=Cfp)

    return dcf, conf_mat, threshold
