import numpy
from numpy import log
import matplotlib.pyplot as plt
import os

from modules.statistics import *


def get_prior_cost_threshold(pi, Cfn=1, Cfp=1):
    return -log((pi * Cfn) / ((1 - pi) * Cfp))


def classify_with_threshold(D, threshold):
    D = vrow(D)

    predictions = numpy.zeros(shape=(D.shape[1]), dtype=numpy.int32)
    predictions[D[0] >= threshold] = 1
    predictions[D[0] < threshold] = 0

    return predictions


def get_dumb_threshold(D_training, D_validation, L, unique_classes, index2class_names=None):
    # If not provided, create a dumb class_name_dictionary
    if index2class_names is None:
        index2class_names = {}
        for class_n in unique_classes:
            index2class_names[class_n] = class_n

    # I need to add this hotfix in order to allow classes of L to be whichever number (not just 0 and 1)
    my_zero = min(unique_classes)
    my_one = max(unique_classes)

    means = get_mean_by_class(D_training, L, unique_classes, index2class_names)  # Size of means will be equal to features dimensionality
    if means[index2class_names[my_one]] < means[index2class_names[my_zero]]:
        D_validation *= -1

    threshold = np.array([0.0])
    for mean in means.values():
        threshold += mean
    threshold /= len(means)

    if threshold.size == 1:
        return threshold[0], D_validation
    else:
        return threshold, D_validation


def try_variating_threshold(D, L, misses, threshold, verbose=False, extremely_verbose=False,
                            plot_name="Thresholds_vs_Misses", folder_name=".", step=0.001, tries=400):
    thresholds_vec = []
    misses_vec = []

    min_misses = misses
    min_threshold = threshold
    min_i = 0

    original_threshold = threshold
    threshold = threshold - step*(tries / 2)

    for i in range(tries):
        threshold += 0.001
        prediction = classify_with_threshold(D, threshold)
        (misses, err_rate) = get_classification_err(L, prediction, how_many_samples(D))

        thresholds_vec.append(threshold)
        misses_vec.append(misses)

        if misses < min_misses:
            min_misses = misses
            min_threshold = threshold
            min_i = i
            if extremely_verbose:
                print(f"{i}) Threshold={threshold}: {misses} misses (over {how_many_samples(D)} samples) "
                      f"detected using LDA, which is a {err_rate}% error rate.")

    plt.figure(num=plot_name)
    plt.xlabel("Threshold")
    plt.ylabel("Misses")
    plt.title(f"{plot_name}")

    plt.plot(thresholds_vec, misses_vec)

    plt.legend(f"Total Samples = {how_many_samples(D)}")
    plt.tight_layout()

    save_folder = f"./{folder_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    full_name_with_path = os.path.join(save_folder, f"./{plot_name}.pdf")

    plt.savefig(full_name_with_path)
    print(f"Saved {plot_name} variating threshold.pdf")
    plt.show()

    if verbose:
        print(f"info:::The minimum was found at try {min_i}, so at threshold: {original_threshold} + {min_i*step} = {min_threshold}")

    return min_threshold, min_misses, min_misses / how_many_samples(D) * 100


""" Returns nof_errors, error_rate, accuracy (all raw, i.e. not multiplied by 100 for the %.
        Remember that this function assumes L and prediction all are for class 0, 1, 2, ... (i.e. 
        starting from zero and with no gaps)"""
def get_classification_err(L, prediction, verbose=False):
    if L.ndim != 1:
        raise ValueError(f"L matrix when calculating error rate should be 1xn, instead is {L.shape}")
    if prediction.ndim != 1:
        raise ValueError(f"prediction matrix when calculating error rate should be 1xn, instead is {prediction.shape}")

    total_samples = prediction.size

    misses = 0
    for i in range(len(L)):
        if verbose:
            print(f"actual: {L[i]}, predicted: {prediction[i]}")
        if L[i] != prediction[i]:
            misses += 1

    rate = misses / total_samples
    accuracy = (len(prediction) - misses) / len(prediction)
    return misses, rate, accuracy


# def get_dcf(predictions, actual_labels, verbose_title=None):
#     if ( set(get_unique_classes(predictions)) > set(get_unique_classes(actual_labels)) ):
#         raise ValueError("Predicted labels should be an (eventually improper) subset of actual labels")
#     if (predictions.size != actual_labels.size):
#         raise ValueError(f"Predictions are {predictions.size} long, while actual labels are {actual_labels.size} long")
#
#     dcf = 0
#     for (prediction, label) in zip(predictions, actual_labels):
#


def get_dcf_binary(confusion_matrix, pi, Cfn, Cfp, verbose=False):
    if confusion_matrix.shape != (2, 2):
        raise ValueError("false positive and negative ratio is defined only for binary tasks, which have 2x2 "
                         "confusion matrices")

    fn_rate, fp_rate = get_fn_fp_rate(confusion_matrix, verbose=verbose)

    fn_tot_cost = pi * Cfn * fn_rate
    fp_tot_cost = (1 - pi) * Cfp * fp_rate

    dcf = fn_tot_cost + fp_tot_cost
    if verbose:
        print(f"DCF is: {dcf:.3f}")

    return dcf


def dumb_classify_two_classes_always_true(L_val):
    if (len(get_unique_classes(L_val)) != 2):
        raise ValueError(f"Tried to perform binary classification with {len(get_unique_classes(L_val))} classes")
    if (L_val.ndim != 1):
        raise ValueError(f"Shape of labels vector should be (size, )")

    return np.ones(L_val.size, dtype=int)


def dumb_classify_two_classes_always_false(L_val):
    if (len(get_unique_classes(L_val)) != 2):
        raise ValueError(f"Tried to perform binary classification with {len(get_unique_classes(L_val))} classes")
    if (L_val.ndim != 1):
        raise ValueError(f"Shape of labels vector should be (size, )")

    return np.zeros(L_val.size, dtype=int)