import sys

from modules.load_datasets import load_iris, split_db_2to1
from modules.mvg_classification import *
from modules.classification import *
from modules.evaluation import *

import matplotlib.pyplot as plt


# # Suppress displaying plots
# def nop():
#     pass
# plt.show = nop

# Print wider lines on console
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(precision=4, suppress=True)


def iris_falsepositives(D, L):
    # Splitting Dataset
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    # print_dataset_info(D_tr, L_tr, "iris - training")
    # print_dataset_info(D_val, L_val, "iris - validation")

    # print("===============================")
    print("Now running MVG")
    logscore, _, _, logscore_posterior = extract_densities_with_MVG(D_tr, L_tr, D_val)
    _, _, _, predictions_MVG = classify_nclasses(logscore_posterior, L_val)
    get_confusion_matrix(predictions_MVG, L_val, verbose_title="MVG")

    print("===============================")
    print("Now running Tied")
    logscore, _, _, logscore_posterior = extract_densities_with_tied(D_tr, L_tr, D_val)
    _, _, _, predictions_tied = classify_nclasses(logscore_posterior, L_val)
    get_confusion_matrix(predictions_tied, L_val, verbose_title="Tied")


def commedia_falsepositives():
    print("Now running divina_commedia")

    lls = np.load("Data/commedia_ll.npy")
    labels = np.load("Data/commedia_labels.npy")

    _, _, _, predictions_MVG = classify_nclasses(lls, labels)
    get_confusion_matrix(predictions_MVG, labels, verbose_title="Divina Commedia")


def optimal_decisions(llrs, labels, verbose=False):

    if verbose:
        print("===============================")

    threshold = get_prior_cost_threshold(pi=0.5, Cfn=1, Cfp=1)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels,
                                    verbose=verbose, verbose_title="pi=0.5, Cfn=1, Cfp=1")
    dcf_05_1_1 = get_dcf_binary(conf_mat, pi=0.5, Cfn=1, Cfp=1)

    if verbose:
        print("===============================")

    threshold = get_prior_cost_threshold(pi=0.8, Cfn=1, Cfp=1)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels,
                                    verbose=verbose, verbose_title="pi=0.8, Cfn=1, Cfp=1")
    dcf_08_1_1 = get_dcf_binary(conf_mat, pi=0.8, Cfn=1, Cfp=1)

    if verbose:
        print("===============================")

    threshold = get_prior_cost_threshold(pi=0.5, Cfn=10, Cfp=1)
    # _, _, _, predictions = opt_classify_two_classes(llrs, labels, pi=0.2, Cfn=10, Cfp=1)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels,
                                    verbose=verbose, verbose_title="pi=0.5, Cfn=10, Cfp=1")
    dcf_05_10_1 = get_dcf_binary(conf_mat, pi=0.5, Cfn=10, Cfp=1)

    if verbose:
        print("===============================")

    threshold = get_prior_cost_threshold(pi=0.8, Cfn=1, Cfp=10)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels,
                                    verbose=verbose, verbose_title="pi=0.8, Cfn=1, Cfp=10")
    dcf_08_1_10 = get_dcf_binary(conf_mat, pi=0.8, Cfn=1, Cfp=10)

    if verbose:
        print("===============================")
        print("Getting DCF scores")
        print( "(π, Cfn, Cfp)   DCFu(B)")
        print(f"(0.5, 1, 1)     {dcf_05_1_1:.3f}")
        print(f"(0.8, 1, 1)     {dcf_08_1_1:.3f}")
        # print(f"(0.2, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.5, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.8, 1, 10)    {dcf_08_1_10:.3f}")

    return dcf_05_1_1, dcf_08_1_1, dcf_05_10_1, dcf_08_1_10


def normalized_dcfs(dcf_05_1_1, dcf_08_1_1, dcf_05_10_1, dcf_08_1_10, verbose=False):
    dcf_05_1_1 = get_normalized_dcf_binary(pi=0.5, Cfn=1, Cfp=1, dcf=dcf_05_1_1)

    dcf_08_1_1 = get_normalized_dcf_binary(pi=0.8, Cfn=1, Cfp=1, dcf=dcf_08_1_1)

    # dcf_05_10_1, _ = get_normalized_dcf_binary(pi=0.2, Cfn=10, Cfp=1)
    dcf_05_10_1 = get_normalized_dcf_binary(pi=0.5, Cfn=10, Cfp=1, dcf=dcf_05_10_1)

    dcf_08_1_10 = get_normalized_dcf_binary(pi=0.8, Cfn=1, Cfp=10, dcf=dcf_08_1_10)

    if verbose:
        print("Getting normalized DCF scores")
        print( "(π, Cfn, Cfp)   DCFu(B)")
        print(f"(0.5, 1, 1)     {dcf_05_1_1:.3f}")
        print(f"(0.8, 1, 1)     {dcf_08_1_1:.3f}")
        # print(f"(0.2, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.5, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.8, 1, 10)    {dcf_08_1_10:.3f}")


def min_dcfs(llrs, labels, verbose=False):

    if verbose:
        print("===============================")
    dcf_05_1_1 = get_min_normalized_dcf_binary(llrs, labels, pi=0.5, Cfn=1, Cfp=1)

    if verbose:
        print("===============================")
    dcf_08_1_1 = get_min_normalized_dcf_binary(llrs, labels, pi=0.8, Cfn=1, Cfp=1)

    if verbose:
        print("===============================")
    dcf_05_10_1 = get_min_normalized_dcf_binary(llrs, labels, pi=0.5, Cfn=10, Cfp=1)

    if verbose:
        print("===============================")
    dcf_08_1_10 = get_min_normalized_dcf_binary(llrs, labels, pi=0.8, Cfn=1, Cfp=10)

    if verbose:
        print("===============================")
        print("Getting DCF scores")
        print( "(π, Cfn, Cfp)   DCFu(B)")
        print(f"(0.5, 1, 1)     {dcf_05_1_1:.3f}")
        print(f"(0.8, 1, 1)     {dcf_08_1_1:.3f}")
        # print(f"(0.2, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.5, 10, 1)    {dcf_05_10_1:.3f}")
        print(f"(0.8, 1, 10)    {dcf_08_1_10:.3f}")

    return dcf_05_1_1, dcf_08_1_1, dcf_05_10_1, dcf_08_1_10


def ROC(llrs, labels, verbose=False):

    pi = 0.5
    Cfn = 1
    Cfp = 1
    plot_ROC(llrs, labels,
             verbose=verbose, verbose_title=f"ROC Curve pi={pi}, Cfn={Cfn}, Cfp={Cfp}")


def get_dcf(pi, Cfn, Cfp, llrs, labels):
    threshold = get_prior_cost_threshold(pi, Cfn, Cfp)
    _, _, _, predictions = opt_classify_two_classes(llrs, labels, threshold)
    conf_mat = get_confusion_matrix(predictions, labels)
    dcf = get_dcf_binary(conf_mat, pi=0.5, Cfn=1, Cfp=1)

    return dcf


def bayes_error_plots(llrs, labels):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)

    normalized_dcfs = []
    min_dcfs = []

    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        dcf = get_dcf(pi, 1, 1, llrs, labels)
        normalized_dcf = get_normalized_dcf_binary(pi, 1, 1, dcf)
        min_dcf = get_min_normalized_dcf_binary(llrs, labels, pi, 1, 1)

        normalized_dcfs.append(normalized_dcf)
        min_dcfs.append(min_dcf)

    plt.figure(num="Bayes Error Plot")
    plt.plot(effPriorLogOdds, normalized_dcfs, label="DCF", color='r')
    plt.plot(effPriorLogOdds, min_dcfs, label="min_DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    plt.title("Bayes Error Plot")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # irisD, irisL = load_iris()
    # iris_falsepositives(irisD, irisL)
    #
    # commedia_falsepositives()


    infparLLRS = np.load('Data/commedia_llr_infpar.npy')
    infparL = np.load('Data/commedia_labels_infpar.npy')

    dcf_05_1_1, dcf_08_1_1, dcf_05_10_1, dcf_08_1_10 = optimal_decisions(infparLLRS, infparL)

    normalized_dcfs(dcf_05_1_1, dcf_08_1_1, dcf_05_10_1, dcf_08_1_10)

    min_dcfs(infparLLRS, infparL, verbose=True)

    ROC(infparLLRS, infparL)

    bayes_error_plots(infparLLRS, infparL)


