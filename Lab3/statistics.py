import numpy as np
from common_matrix_operations import *


def how_many_classes(Labels):
    unique_values = np.unique(Labels)
    return len(unique_values)


def get_unique_classes(Labels):
    return np.unique(Labels)


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


def how_many_features(Data):
    return Data.shape[0]


def how_many_samples(Data):
    return Data.shape[1]


def print_dataset_info(Data, Labels, name):
    msg = f"info:::The dataset [{name}] contains {how_many_samples(Data)} samples," \
          f" and has {how_many_features(Data)} features" \
          f" and {how_many_classes(Labels)} classes"
    print(msg)


