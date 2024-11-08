from modules.classification import get_prior_cost_threshold, get_dcf_binary
from modules.mvg_classification import opt_classify_two_classes
from modules.statistics import get_confusion_matrix


def get_dcf(pi, Cfn, Cfp, llrs, labels):
    """ This function is used by plot_bayes_error_plots.
        It should probably be in another module, but I don't have time to find where """
    threshold = get_prior_cost_threshold(pi, Cfn, Cfp)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels)
    dcf = get_dcf_binary(conf_mat, pi=0.5, Cfn=1, Cfp=1)

    return dcf