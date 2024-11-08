import numpy as np
from modules.common_matrix_operations import *


def how_many_classes(Labels):
    unique_values = np.unique(Labels)
    return len(unique_values)


def how_many_features(Data):
    return Data.shape[0]


def how_many_samples(Data):
    return Data.shape[1]


""" Returns a list with the unique labels """
def get_unique_classes(Labels):
    return np.unique(Labels).tolist()


def get_mean(D):
    return D.mean(1)


def get_mean_by_class(D, L, unique_classes, index2class_name=None):
    # If not provided, create a dumb class_name_dictionary
    if index2class_name is None:
        index2class_name = {}
        for class_n in unique_classes:
            index2class_name[class_n] = class_n

    means = {}
    for class_n in unique_classes:
        D_of_class_x = D[:, L == class_n]
        mu = D_of_class_x.mean(1)

        class_name = index2class_name[class_n]
        if mu.size == 1:
            means[class_name] = mu
        else:
            means[class_name] = onedim_arr_to_colvector(mu)

    return means


def get_covariance_matrix(Data):
    mu = Data.mean(1).reshape((Data.shape[0], 1))
    Cov = ((Data - mu) @ (Data - mu).T) / float(Data.shape[1])

    return Cov


def get_covariance_matrix_by_class(D, L):
    unique_classes = get_unique_classes(L)

    covariance_matrices = [0 for _ in range(max(unique_classes) + 1)]

    for class_x in unique_classes:
        D_of_class_x = D[:, L == class_x]
        cov_of_class_x = get_covariance_matrix(D_of_class_x)
        covariance_matrices[class_x] = cov_of_class_x

    return covariance_matrices


def get_within_class_covariance_matrix(D, L):
    unique_classes = get_unique_classes(L)

    within_class_covariance = 0
    for class_x in unique_classes:
        D_of_class_x = D[:, L == class_x]
        cov_of_class_x = get_covariance_matrix(D_of_class_x)

        nof_samples_of_class_x = how_many_samples(D_of_class_x)
        within_class_covariance += (nof_samples_of_class_x * cov_of_class_x)
    within_class_covariance = (1 / float(D.shape[1])) * within_class_covariance

    return within_class_covariance


def get_correlation_matrix(C):
    correlation_matrix = C / (onedim_arr_to_colvector(C.diagonal() ** 0.5) * onedim_arr_to_rowvector(C.diagonal() ** 0.5))
    return correlation_matrix


def print_dataset_info(Data, Labels, name):
    msg = f"info:::The dataset [{name}] contains {how_many_samples(Data)} samples," \
          f" and has {how_many_features(Data)} features" \
          f" and {how_many_classes(Labels)} classes"
    print(msg)


def get_confusion_matrix(predictions, actual_labels, verbose=False, verbose_title=None):
    if ( set(get_unique_classes(predictions)) > set(get_unique_classes(actual_labels)) ):
        raise ValueError("Predicted labels should be an (eventually improper) subset of actual labels")
    if (predictions.size != actual_labels.size):
        raise ValueError(f"Predictions are {predictions.size} long, while actual labels are {actual_labels.size} long")

    nof_classes = how_many_classes(actual_labels)
    confusion_matrix = np.zeros((nof_classes, nof_classes), dtype=int)

    for v1, v2 in zip(predictions, actual_labels):
        confusion_matrix[v1, v2] += 1

    if verbose:
        print(f"Confusion matrix for {verbose_title} is:")
        print(confusion_matrix)

    return confusion_matrix


""" If the confusion matrix is 2x2 (i.e. binary classification), returns two values:
    (false positive ratio, false negative ratio) """
def get_fn_fp_rate(confusion_matrix, verbose=False):
    if confusion_matrix.shape != (2, 2):
        raise ValueError("false positive and negative ratio is defined only for binary tasks, which have 2x2 "
                         "confusion matrices")

    fp_rate = confusion_matrix[1, 0] / (confusion_matrix[1, 0] + confusion_matrix[0, 0])
    fn_rate = confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])

    if verbose:
        print(f"FP and FN ratios are: FN = {fn_rate:.2f} FP = {fp_rate:.2f}")

    return fn_rate, fp_rate


def how_many_foreach_class(labels):
    unique_classes = get_unique_classes(labels)

    my_dict = dict()

    for c in unique_classes:
        my_dict[c] = 0

    for label in labels:
        my_dict[label] += 1

    return my_dict

