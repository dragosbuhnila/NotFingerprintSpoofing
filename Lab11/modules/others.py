from modules.classification import get_prior_cost_threshold, get_dcf_binary
from modules.mvg_classification import opt_classify_two_classes
from modules.statistics import get_confusion_matrix


def get_dcf(pi, Cfn, Cfp, llrs, labels):
    """ Wrapper function to only get dcf from get_dcf__includes_classification(...) -> dcf, conf_mat, threshold """
    dcf, _, _ = get_dcf__includes_classification(llrs, labels, pi, Cfn, Cfp)
    return dcf

