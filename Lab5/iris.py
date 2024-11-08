from modules.classification import get_dumb_threshold, classify_with_threshold, get_classification_err
from modules.load_datasets import load_iris, split_db_2to1
from modules.projections import get_LDA_projection_matrix
from modules.statistics import *
from modules.mvg_classification import *


import scipy

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))


def compute_ML_estimates(D, L, verbose=False):
    unique_classes = get_unique_classes(L)

    for class_x in unique_classes:
        D_of_class_x = D[:, L == class_x]
        mu_of_class_x = get_mean(D_of_class_x)
        cov_of_class_x = get_covariance_matrix(D_of_class_x)

        if verbose:
            print("------------------------")
            print(f"Mean of class {class_x} is:")
            print(mu_of_class_x)
            print(f"Covariance Matrix of class {class_x} is:")
            print(cov_of_class_x)

    # return means, covariances


if __name__ == "__main__":
    D_full, L_full = load_iris()
    # Removing Setosa
    D = D_full[:, L_full != 0]
    L = L_full[L_full != 0]

    # Splitting Dataset
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    print_dataset_info(D_tr, L_tr, "iris - training")
    print_dataset_info(D_val, L_val, "iris - validation")

    compute_ML_estimates(D_tr, L_tr)

    # print("===============================")
    # print("Now running MVG")
    logscore, _, _, logscore_posterior = extract_densities_with_MVG(D_tr, L_tr, D_val)
    # classify_nclasses(logscore_posterior, L_val, verbose=True)
    # classify_two_classes(logscore, L_val, verbose=True)
    #
    # print("===============================")
    # print("Now running Naive")
    # logscore, _, _, logscore_posterior = extract_densities_with_naive(D_tr, L_tr, D_val)
    # classify_nclasses(logscore_posterior, L_val, verbose=True)
    # classify_two_classes(logscore, L_val, verbose=True)

    print("===============================")
    print("Now running Tied")
    logscore, _, _, logscore_posterior = extract_densities_with_tied(D_tr, L_tr, D_val)
    classify_nclasses(logscore_posterior, L_val, verbose=True)
    classify_two_classes(logscore, L_val, verbose=True)
    classify_over_LDA(D_tr, L_tr, D_val, L_val, verbose=True)

