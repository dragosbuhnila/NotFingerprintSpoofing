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




